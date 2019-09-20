# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2016, 2017 Matt Wala
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION
import numpy as np
import pyopencl as cl

from pytools import memoize_method
from boxtree.area_query import AreaQueryElementwiseTemplate
from boxtree.tools import InlineBinarySearch
from pytential.qbx.utils import (
        QBX_TREE_C_PREAMBLE, QBX_TREE_MAKO_DEFS, TreeWranglerBase,
        TreeCodeContainerMixin)

from pytools import ProcessLogger, log_process

import logging
logger = logging.getLogger(__name__)


# max_levels granularity for the stack used by the tree descent code in the
# area query kernel.
MAX_LEVELS_INCREMENT = 10


__doc__ = """
The refiner takes a layer potential source and refines it until it satisfies
three global QBX refinement criteria:

   * *Condition 1* (Expansion disk undisturbed by sources)
      A center must be closest to its own source.

   * *Condition 2* (Sufficient quadrature sampling from all source panels)
      The quadrature contribution from each panel is as accurate
      as from the center's own source panel.

   * *Condition 3* (Panel size bounded based on kernel length scale)
      The panel size is bounded by a kernel length scale. This
      applies only to Helmholtz kernels.

Warnings emitted by refinement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RefinerNotConvergedWarning

Helper functions
^^^^^^^^^^^^^^^^

.. autofunction:: make_empty_refine_flags

Refiner driver
^^^^^^^^^^^^^^

.. autoclass:: RefinerCodeContainer

.. autoclass:: RefinerWrangler

.. automethod:: refine_qbx_stage1
.. automethod:: refine_qbx_stage2
"""

# {{{ kernels

# Refinement checker for Condition 1.
EXPANSION_DISK_UNDISTURBED_BY_SOURCES_CHECKER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_source_starts,
        particle_id_t *box_to_source_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t *panel_to_center_starts,
        particle_id_t source_offset,
        particle_id_t center_offset,
        particle_id_t *sorted_target_ids,
        coord_t *center_danger_zone_radii,
        coord_t expansion_disturbance_tolerance,
        int npanels,

        /* output */
        int *panel_refine_flags,
        int *found_panel_to_refine,

        /* input, dim-dependent length */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
        """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        particle_id_t icenter = i;

        ${load_particle("INDEX_FOR_CENTER_PARTICLE(icenter)", ball_center)}
        ${ball_radius} = (1-expansion_disturbance_tolerance)
                    * center_danger_zone_radii[icenter];
        """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        /* Check that each source in the leaf box is sufficiently far from the
           center; if not, mark the panel for refinement. */

        for (particle_id_t source_idx = box_to_source_starts[${leaf_box_id}];
             source_idx < box_to_source_starts[${leaf_box_id} + 1];
             ++source_idx)
        {
            particle_id_t source = box_to_source_lists[source_idx];
            particle_id_t source_panel = bsearch(
                panel_to_source_starts, npanels + 1, source);

            /* Find the panel associated with this center. */
            particle_id_t center_panel = bsearch(panel_to_center_starts, npanels + 1,
                icenter);

            coord_vec_t source_coords;
            ${load_particle("INDEX_FOR_SOURCE_PARTICLE(source)", "source_coords")}

            bool is_close = (
                distance(${ball_center}, source_coords)
                <= (1-expansion_disturbance_tolerance)
                        * center_danger_zone_radii[icenter]);

            if (is_close)
            {
                panel_refine_flags[center_panel] = 1;
                *found_panel_to_refine = 1;
                break;
            }
        }
        """,
    name="check_center_closest_to_orig_panel",
    preamble=str(InlineBinarySearch("particle_id_t")))


# Refinement checker for Condition 2.
SUFFICIENT_SOURCE_QUADRATURE_RESOLUTION_CHECKER = AreaQueryElementwiseTemplate(
    extra_args=r"""
        /* input */
        particle_id_t *box_to_center_starts,
        particle_id_t *box_to_center_lists,
        particle_id_t *panel_to_source_starts,
        particle_id_t source_offset,
        particle_id_t center_offset,
        particle_id_t *sorted_target_ids,
        coord_t *source_danger_zone_radii_by_panel,
        int npanels,

        /* output */
        int *panel_refine_flags,
        int *found_panel_to_refine,

        /* input, dim-dependent length */
        %for ax in AXIS_NAMES[:dimensions]:
            coord_t *particles_${ax},
        %endfor
        """,
    ball_center_and_radius_expr=QBX_TREE_C_PREAMBLE + QBX_TREE_MAKO_DEFS + r"""
        /* Find the panel associated with this source. */
        particle_id_t my_panel = bsearch(panel_to_source_starts, npanels + 1, i);

        ${load_particle("INDEX_FOR_SOURCE_PARTICLE(i)", ball_center)}
        ${ball_radius} = source_danger_zone_radii_by_panel[my_panel];
        """,
    leaf_found_op=QBX_TREE_MAKO_DEFS + r"""
        /* Check that each center in the leaf box is sufficiently far from the
           panel; if not, mark the panel for refinement. */

        for (particle_id_t center_idx = box_to_center_starts[${leaf_box_id}];
             center_idx < box_to_center_starts[${leaf_box_id} + 1];
             ++center_idx)
        {
            particle_id_t center = box_to_center_lists[center_idx];

            coord_vec_t center_coords;
            ${load_particle(
                "INDEX_FOR_CENTER_PARTICLE(center)", "center_coords")}

            bool is_close = (
                distance(${ball_center}, center_coords)
                <= source_danger_zone_radii_by_panel[my_panel]);

            if (is_close)
            {
                panel_refine_flags[my_panel] = 1;
                *found_panel_to_refine = 1;
                break;
            }
        }
        """,
    name="check_source_quadrature_resolution",
    preamble=str(InlineBinarySearch("particle_id_t")))

# }}}


# {{{ code container

class RefinerCodeContainer(TreeCodeContainerMixin):

    def __init__(self, cl_context, tree_code_container):
        self.cl_context = cl_context
        self.tree_code_container = tree_code_container

    @memoize_method
    def expansion_disk_undisturbed_by_sources_checker(
            self, dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
            particle_id_dtype, max_levels):
        return EXPANSION_DISK_UNDISTURBED_BY_SOURCES_CHECKER.generate(
                self.cl_context,
                dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def sufficient_source_quadrature_resolution_checker(
            self, dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
            particle_id_dtype, max_levels):
        return SUFFICIENT_SOURCE_QUADRATURE_RESOLUTION_CHECKER.generate(
                self.cl_context,
                dimensions, coord_dtype, box_id_dtype, peer_list_idx_dtype,
                max_levels,
                extra_type_aliases=(("particle_id_t", particle_id_dtype),))

    @memoize_method
    def element_prop_threshold_checker(self):
        knl = lp.make_kernel(
            "{[ielement]: 0<=ielement<nelements}",
            """
            for ielement
                <> over_threshold = element_property[ielement] > threshold
                if over_threshold
                    refine_flags[ielement] = 1
                    refine_flags_updated = 1 {id=write_refine_flags_updated}
                end
            end
            """,
            options="return_dict",
            silenced_warnings="write_race(write_refine_flags_updated)",
            name="refine_kernel_length_scale_to_quad_resolution_ratio",
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        knl = lp.split_iname(knl, "ielement", 128, inner_tag="l.0", outer_tag="g.0")
        return knl

    def get_wrangler(self, queue):
        """
        :arg queue:
        """
        return RefinerWrangler(self, queue)

# }}}


# {{{ wrangler

class RefinerWrangler(TreeWranglerBase):
    # {{{ check subroutines for conditions 1-3

    @log_process(logger)
    def check_expansion_disks_undisturbed_by_sources(self,
            stage1_density_discr, tree, peer_lists,
            expansion_disturbance_tolerance,
            refine_flags,
            debug, wait_for=None):

        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = MAX_LEVELS_INCREMENT * div_ceil(
                tree.nlevels, MAX_LEVELS_INCREMENT)

        knl = self.code_container.expansion_disk_undisturbed_by_sources_checker(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)

        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        found_panel_to_refine = cl.array.zeros(self.queue, 1, np.int32)
        found_panel_to_refine.finish()
        unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

        from pytential import bind, sym
        center_danger_zone_radii = bind(stage1_density_discr,
                sym.expansion_radii(stage1_density_discr.ambient_dim,
                    granularity=sym.GRANULARITY_CENTER))(self.queue)

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_source_starts,
                tree.box_to_qbx_source_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_panel_to_center_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_center_slice.start,
                tree.sorted_target_ids,
                center_danger_zone_radii,
                expansion_disturbance_tolerance,
                tree.nqbxpanels,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=slice(tree.nqbxcenters),
            queue=self.queue,
            wait_for=wait_for)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        return found_panel_to_refine.get()[0] == 1

    @log_process(logger)
    def check_sufficient_source_quadrature_resolution(self,
            stage2_density_discr, tree, peer_lists, refine_flags,
            debug, wait_for=None):

        # Avoid generating too many kernels.
        from pytools import div_ceil
        max_levels = MAX_LEVELS_INCREMENT * div_ceil(
                tree.nlevels, MAX_LEVELS_INCREMENT)

        knl = self.code_container.sufficient_source_quadrature_resolution_checker(
                tree.dimensions,
                tree.coord_dtype, tree.box_id_dtype,
                peer_lists.peer_list_starts.dtype,
                tree.particle_id_dtype,
                max_levels)
        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        found_panel_to_refine = cl.array.zeros(self.queue, 1, np.int32)
        found_panel_to_refine.finish()

        from pytential import bind, sym
        dd = sym.as_dofdesc(sym.GRANULARITY_ELEMENT).to_stage2()
        source_danger_zone_radii_by_panel = bind(stage2_density_discr,
                sym._source_danger_zone_radii(
                    stage2_density_discr.ambient_dim, dofdesc=dd))(self.queue)
        unwrap_args = AreaQueryElementwiseTemplate.unwrap_args

        evt = knl(
            *unwrap_args(
                tree, peer_lists,
                tree.box_to_qbx_center_starts,
                tree.box_to_qbx_center_lists,
                tree.qbx_panel_to_source_starts,
                tree.qbx_user_source_slice.start,
                tree.qbx_user_center_slice.start,
                tree.sorted_target_ids,
                source_danger_zone_radii_by_panel,
                tree.nqbxpanels,
                refine_flags,
                found_panel_to_refine,
                *tree.sources),
            range=slice(tree.nqbxsources),
            queue=self.queue,
            wait_for=wait_for)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        return found_panel_to_refine.get()[0] == 1

    def check_element_prop_threshold(self, element_property, threshold, refine_flags,
            debug, wait_for=None):
        knl = self.code_container.element_prop_threshold_checker()

        if debug:
            npanels_to_refine_prev = cl.array.sum(refine_flags).get()

        evt, out = knl(self.queue,
                       element_property=element_property,
                       refine_flags=refine_flags,
                       refine_flags_updated=np.array(0),
                       threshold=np.array(threshold),
                       wait_for=wait_for)

        cl.wait_for_events([evt])

        if debug:
            npanels_to_refine = cl.array.sum(refine_flags).get()
            if npanels_to_refine > npanels_to_refine_prev:
                logger.debug("refiner: found {} panel(s) to refine".format(
                    npanels_to_refine - npanels_to_refine_prev))

        return (out["refine_flags_updated"].get() == 1).all()

    # }}}

    def refine(self, density_discr, refiner, refine_flags, factory, debug):
        """
        Refine the underlying mesh and discretization.
        """
        if isinstance(refine_flags, cl.array.Array):
            refine_flags = refine_flags.get(self.queue)
        refine_flags = refine_flags.astype(np.bool)

        with ProcessLogger(logger, "refine mesh"):
            refiner.refine(refine_flags)
            from meshmode.discretization.connection import make_refinement_connection
            conn = make_refinement_connection(refiner, density_discr, factory)

        return conn

# }}}


class RefinerNotConvergedWarning(UserWarning):
    pass


def make_empty_refine_flags(queue, density_discr):
    """Return an array on the device suitable for use as element refine flags.

    :arg queue: An instance of :class:`pyopencl.CommandQueue`.
    :arg lpot_source: An instance of :class:`QBXLayerPotentialSource`.

    :returns: A :class:`pyopencl.array.Array` suitable for use as refine flags,
        initialized to zero.
    """
    result = cl.array.zeros(queue, density_discr.mesh.nelements, np.int32)
    result.finish()
    return result


# {{{ main entry point

def _warn_max_iterations(violated_criteria, expansion_disturbance_tolerance):
    from warnings import warn
    warn(
            "QBX layer potential source refiner did not terminate "
            "after %d iterations (the maximum). "
            "You may pass 'visualize=True' to with_refinement() "
            "to see what area of the geometry is causing trouble. "
            "If the issue is disturbance of expansion disks, you may "
            "pass a slightly increased value (currently: %g) for "
            "_expansion_disturbance_tolerance in with_refinement(). "
            "As a last resort, "
            "you may use Python's warning filtering mechanism to "
            "not treat this warning as an error. "
            "The criteria triggering refinement in each iteration "
            "were: %s. " % (
                len(violated_criteria),
                expansion_disturbance_tolerance,
                ", ".join(
                    "%d: %s" % (i+1, vc_text)
                    for i, vc_text in enumerate(violated_criteria))),
            RefinerNotConvergedWarning)


def _visualize_refinement(queue, source_name, discr,
        niter, stage_nr, stage_name, flags):
    if stage_nr not in (1, 2):
        raise ValueError("unexpected stage number")

    flags = flags.get()
    logger.info("for stage %s: splitting %d/%d stage-%d elements",
            stage_name, np.sum(flags), discr.mesh.nelements, stage_nr)

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(queue, discr, 3)

    assert len(flags) == discr.mesh.nelements

    flags = flags.astype(np.bool)
    nodes_flags = np.zeros(discr.nnodes)
    for grp in discr.groups:
        meg = grp.mesh_el_group
        grp.view(nodes_flags)[
                flags[meg.element_nr_base:meg.nelements+meg.element_nr_base]] = 1

    nodes_flags = cl.array.to_device(queue, nodes_flags)
    vis_data = [
        ("refine_flags", nodes_flags),
        ]

    if 0:
        from pytential import sym, bind
        bdry_normals = bind(discr, sym.normal(discr.ambient_dim))(
                queue).as_vector(dtype=object)
        vis_data.append(("bdry_normals", bdry_normals),)

    if isinstance(source_name, type):
        source_name = source_name.__name__
    source_name = str(source_name).lower().replace('_', '-').replace('/', '-')

    vis.write_vtk_file("refinement-%s-%s-%03d.vtu" %
            (source_name, stage_name, niter),
            vis_data, overwrite=True)


def _make_quad_stage2_discr(lpot_source, stage2_density_discr):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            QuadratureSimplexGroupFactory

    return Discretization(
            lpot_source.cl_context,
            stage2_density_discr.mesh,
            QuadratureSimplexGroupFactory(lpot_source.fine_order),
            lpot_source.real_dtype)


def refine_qbx_stage1(places, source_name,
        wrangler, group_factory,
        kernel_length_scale=None,
        scaled_max_curvature_threshold=None,
        expansion_disturbance_tolerance=None,
        maxiter=None, refiner=None, debug=None, visualize=False):
    """Stage 1 refinement entry point.

    :arg places: a :class:`~pytential.symbolic.geometry.GeometryCollection`.
    :arg source_name: symbolic name for the layer potential to be refined.
    :arg wrangler: a :class:`RefinerWrangler`.
    :arg group_factory: a :class:`~meshmode.discretization.ElementGroupFactory`.

    :arg kernel_length_scale: The kernel length scale, or *None* if not
        applicable. All panels are refined to below this size.
    :maxiter: maximum number of refinement iterations.

    :returns: a tuple of ``(discr, conn)``, where ``discr`` is the refined
        discretizations and ``conn`` is a
        :class:`meshmode.discretization.connection.DiscretizationConnection`
        going from the original
        :attr:`pytential.source.LayerPotentialSourceBase.density_discr`
        refined discretization.
    """
    from pytential import sym
    source_name = sym.as_dofdesc(source_name).geometry
    lpot_source = places.get_geometry(source_name)
    density_discr = lpot_source.density_discr

    if maxiter is None:
        maxiter = 10

    if debug is None:
        debug = lpot_source.debug

    if expansion_disturbance_tolerance is None:
        expansion_disturbance_tolerance = 0.025

    if refiner is None:
        from meshmode.mesh.refinement import RefinerWithoutAdjacency
        refiner = RefinerWithoutAdjacency(density_discr.mesh)

    connections = []
    violated_criteria = []
    iter_violated_criteria = ["start"]
    niter = 0

    stage1_density_discr = density_discr

    queue = wrangler.queue
    while iter_violated_criteria:
        iter_violated_criteria = []
        niter += 1

        if niter > maxiter:
            _warn_max_iterations(violated_criteria,
                    expansion_disturbance_tolerance)
            break

        refine_flags = make_empty_refine_flags(queue,
                stage1_density_discr)

        if kernel_length_scale is not None:
            with ProcessLogger(logger,
                    "checking kernel length scale to panel size ratio"):

                from pytential import bind
                quad_resolution = bind(stage1_density_discr,
                        sym._quad_resolution(stage1_density_discr.ambient_dim,
                            dofdesc=sym.GRANULARITY_ELEMENT))(queue)

                violates_kernel_length_scale = \
                        wrangler.check_element_prop_threshold(
                                element_property=quad_resolution,
                                threshold=kernel_length_scale,
                                refine_flags=refine_flags, debug=debug)

                if violates_kernel_length_scale:
                    iter_violated_criteria.append("kernel length scale")
                    if visualize:
                        _visualize_refinement(queue, source_name,
                                stage1_density_discr,
                                niter, 1, "kernel-length-scale", refine_flags)

        if scaled_max_curvature_threshold is not None:
            with ProcessLogger(logger,
                    "checking scaled max curvature threshold"):
                from pytential import bind
                scaled_max_curv = bind(stage1_density_discr,
                    sym.ElementwiseMax(
                        sym._scaled_max_curvature(stage1_density_discr.ambient_dim),
                        dofdesc=sym.GRANULARITY_ELEMENT))(queue)

                violates_scaled_max_curv = \
                        wrangler.check_element_prop_threshold(
                                element_property=scaled_max_curv,
                                threshold=scaled_max_curvature_threshold,
                                refine_flags=refine_flags, debug=debug)

                if violates_scaled_max_curv:
                    iter_violated_criteria.append("curvature")
                    if visualize:
                        _visualize_refinement(queue, source_name,
                                stage1_density_discr,
                                niter, 1, "curvature", refine_flags)

        if not iter_violated_criteria:
            # Only start building trees once the simple length-based criteria
            # are happy.

            stage1_places = places.copy({
                (source_name, sym.QBX_SOURCE_STAGE1): stage1_density_discr
                })

            # Build tree and auxiliary data.
            # FIXME: The tree should not have to be rebuilt at each iteration.
            tree = wrangler.build_tree(stage1_places,
                    sources_list=[source_name])
            peer_lists = wrangler.find_peer_lists(tree)

            has_disturbed_expansions = \
                    wrangler.check_expansion_disks_undisturbed_by_sources(
                            stage1_density_discr, tree, peer_lists,
                            expansion_disturbance_tolerance,
                            refine_flags, debug)
            if has_disturbed_expansions:
                iter_violated_criteria.append("disturbed expansions")
                if visualize:
                    _visualize_refinement(queue, source_name,
                            stage1_density_discr,
                            niter, 1, "disturbed-expansions", refine_flags)

            del tree
            del peer_lists

        if iter_violated_criteria:
            violated_criteria.append(
                    " and ".join(iter_violated_criteria))

            conn = wrangler.refine(
                    stage1_density_discr, refiner, refine_flags,
                    group_factory, debug)
            stage1_density_discr = conn.to_discr
            connections.append(conn)

        del refine_flags

    from meshmode.discretization.connection import ChainedDiscretizationConnection
    conn = ChainedDiscretizationConnection(connections,
            from_discr=density_discr)

    return stage1_density_discr, conn


def refine_qbx_stage2(places, source_name,
        wrangler, group_factory,
        expansion_disturbance_tolerance=None,
        force_stage2_uniform_refinement_rounds=None,
        maxiter=None, refiner=None,
        debug=None, visualize=False):
    """Stage 1 refinement entry point.

    :arg places: a :class:`~pytential.symbolic.geometry.GeometryCollection`.
    :arg source_name: symbolic name for the layer potential to be refined.
    :arg wrangler: a :class:`RefinerWrangler`.
    :arg group_factory: a :class:`~meshmode.discretization.ElementGroupFactory`.

    :maxiter: maximum number of refinement iterations.

    :returns: a tuple of ``(discr, conn)``, where ``discr`` is the refined
        discretizations and ``conn`` is a
        :class:`meshmode.discretization.connection.DiscretizationConnection`
        going from the stage 1 discretization (see :func:`refine_qbx_stage1`)
        to ``discr``.
    """
    from pytential import sym
    source_name = sym.as_dofdesc(source_name).geometry

    lpot_source = places.get_geometry(source_name)
    stage1_density_discr = places.get_discretization(
            sym.as_dofdesc(source_name).to_stage1())

    if maxiter is None:
        maxiter = 10

    if debug is None:
        debug = lpot_source.debug

    if expansion_disturbance_tolerance is None:
        expansion_disturbance_tolerance = 0.025

    if force_stage2_uniform_refinement_rounds is None:
        force_stage2_uniform_refinement_rounds = 0

    if refiner is None:
        from meshmode.mesh.refinement import RefinerWithoutAdjacency
        refiner = RefinerWithoutAdjacency(stage1_density_discr.mesh)

    connections = []
    violated_criteria = []
    iter_violated_criteria = ["start"]
    niter = 0

    stage2_density_discr = stage1_density_discr

    queue = wrangler.queue
    while iter_violated_criteria:
        iter_violated_criteria = []
        niter += 1

        if niter > maxiter:
            _warn_max_iterations(violated_criteria,
                    expansion_disturbance_tolerance)
            break

        stage2_places = places.copy({
            (source_name, sym.QBX_SOURCE_STAGE1): stage1_density_discr,
            (source_name, sym.QBX_SOURCE_STAGE2): stage2_density_discr,
            (source_name, sym.QBX_SOURCE_QUAD_STAGE2):
                _make_quad_stage2_discr(lpot_source, stage2_density_discr)
            })

        # Build tree and auxiliary data.
        # FIXME: The tree should not have to be rebuilt at each iteration.
        tree = wrangler.build_tree(stage2_places,
                sources_list=[source_name],
                use_stage2_discr=True)
        peer_lists = wrangler.find_peer_lists(tree)
        refine_flags = make_empty_refine_flags(queue, stage2_density_discr)

        has_insufficient_quad_resolution = \
                wrangler.check_sufficient_source_quadrature_resolution(
                        stage2_density_discr, tree, peer_lists, refine_flags,
                        debug)
        if has_insufficient_quad_resolution:
            iter_violated_criteria.append("insufficient quadrature resolution")
            if visualize:
                _visualize_refinement(queue, source_name,
                        stage2_density_discr,
                        niter, 2, "quad-resolution", refine_flags)

        if iter_violated_criteria:
            violated_criteria.append(" and ".join(iter_violated_criteria))

            conn = wrangler.refine(stage2_density_discr,
                    refiner, refine_flags, group_factory,
                    debug)
            stage2_density_discr = conn.to_discr
            connections.append(conn)

        del tree
        del refine_flags
        del peer_lists

    for round in range(force_stage2_uniform_refinement_rounds):
        conn = wrangler.refine(
                stage2_density_discr,
                refiner,
                np.ones(stage2_density_discr.mesh.nelements, dtype=np.bool),
                group_factory, debug)
        stage2_density_discr = conn.to_discr
        connections.append(conn)

    from meshmode.discretization.connection import ChainedDiscretizationConnection
    conn = ChainedDiscretizationConnection(connections,
            from_discr=stage1_density_discr)

    return stage2_density_discr, conn

# }}}

# vim: foldmethod=marker:filetype=pyopencl
