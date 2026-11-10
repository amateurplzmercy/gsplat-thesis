"""Common render settings for the full and streamed baselines."""

from gsplat import rasterization

NEAR = 0.01
FAR = 1e10
EPS2D = 0.3


def render_scene(inputs, view, intrinsics, width, height, degree):
    return rasterization(**inputs, viewmats=view, Ks=intrinsics,
                         width=width, height=height, sh_degree=degree,
                         near_plane=NEAR, far_plane=FAR, eps2d=EPS2D,
                         packed=True, radius_clip=0.0, rasterize_mode="classic")
