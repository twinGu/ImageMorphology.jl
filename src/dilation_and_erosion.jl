# Erode and dilate support 3x3 regions only (and higher-dimensional generalizations).

"""
```
imgd = dilate(img, [region])
```

perform a max-filter over nearest-neighbors. The
default is 8-connectivity in 2d, 27-connectivity in 3d, etc. You can specify the
list of dimensions that you want to include in the connectivity, e.g., `region =
[1,2]` would exclude the third dimension from filtering.
"""
dilate(img::AbstractArray, region=coords_spatial(img); dist=1) = dilate!(copy(img), region, dist=dist)

"""
```
imge = erode(img, [region])
```

perform a min-filter over nearest-neighbors. The
default is 8-connectivity in 2d, 27-connectivity in 3d, etc. You can specify the
list of dimensions that you want to include in the connectivity, e.g., `region =
[1,2]` would exclude the third dimension from filtering.
"""
erode(img::AbstractArray, region=coords_spatial(img); dist=1) = erode!(copy(img), region, dist=dist)

dilate!(maxfilt, region=coords_spatial(maxfilt); dist=1) = extremefilt!(maxfilt, max, region, dist=dist)
erode!(minfilt, region=coords_spatial(minfilt); dist=1) = extremefilt!(minfilt, min, region, dist=dist)
function extremefilt!(A::AbstractArray, select::Function, region=coords_spatial(A); dist=1)
    dist = abs(dist)
    inds = axes(A)
    for d = 1:ndims(A)
        if size(A, d) == 1 || !in(d, region)
            continue
        end
        Rpre = CartesianIndices(inds[1:d-1])
        Rpost = CartesianIndices(inds[d+1:end])
        _extremefilt!(A, select, Rpre, inds[d], Rpost, dist)
    end
    A
end

@noinline function _extremefilt!(A, select, Rpre, inds, Rpost)
    # TODO: improve the cache efficiency
    for Ipost in Rpost, Ipre in Rpre
        # first element along dim
        i1 = first(inds)
        a2, a3 = A[Ipre, i1, Ipost], A[Ipre, i1+1, Ipost]
        A[Ipre, i1, Ipost] = select(a2, a3)
        # interior along dim
        for i = i1+2:last(inds)
            a1, a2 = a2, a3
            a3 = A[Ipre, i, Ipost]
            A[Ipre, i-1, Ipost] = select(select(a1, a2), a3)
        end
        # last element along dim
        A[Ipre, last(inds), Ipost] = select(a2, a3)
    end
    A
end

# @noinline function _extremefilt!(A, select, Rpre, inds, Rpost)
#     # TODO: improve the cache efficiency
#     for Ipost in Rpost, Ipre in Rpre
#         # first element along dim
#         i1 = first(inds)
#         a2, a3, a4, a5 = A[Ipre, i1, Ipost], A[Ipre, i1+1, Ipost], A[Ipre, i1+2, Ipost], A[Ipre, i1+3, Ipost]
#         A[Ipre, i1, Ipost] = select(a2, a3, a4)
#         # second element along dim
#         A[Ipre, i1+1, Ipost] = select(a2, a3, a4, a5)
#         # interior along dim
#         for i = i1+4:last(inds)
#             a1, a2, a3, a4 = a2, a3, a4, a5
#             a5 = A[Ipre, i, Ipost]
#             A[Ipre, i-2, Ipost] = select(select(select(select(a1, a2), a3), a4), a5)
#         end
#         # second-to-last element along dim
#
#         A[Ipre, last(inds)-1, Ipost] = select(a2, a3, a4, a5)
#         # last element along dim
#         A[Ipre, last(inds), Ipost] = select(a3, a4, a5)
#     end
#     A
# end

@noinline function _extremefilt!(A, select, Rpre, inds, Rpost, dist)
    # TODO: improve the cache efficiency
    for Ipost in Rpost, Ipre in Rpre
        i1 = first(inds)
        a = zeros(eltype(A), dist*2+1)
        a[2:end] = [A[Ipre, i1+d, Ipost] for d in 0:(2*dist)-1]
        for d in 1:dist
            A[Ipre, i1+d-1, Ipost] = select(a[2:end-dist+d]...)
        end
        for i = i1+2*dist:last(inds)
            a[1:end-1] = a[2:end]
            a[end] = A[Ipre, i, Ipost]
            A[Ipre, i-dist, Ipost] = select(a...)
        end
        for d in dist:-1:1
            A[Ipre, last(inds)-d+1, Ipost] = select(a[2+dist-d:end]...)
        end
    end
    A
end

"""
`imgo = opening(img, [region])` performs the `opening` morphology operation, equivalent to `dilate(erode(img))`.
`region` allows you to control the dimensions over which this operation is performed.
"""
opening(img::AbstractArray, region=coords_spatial(img); dist=1) = opening!(copy(img), region, dist=dist)
opening!(img::AbstractArray, region=coords_spatial(img); dist=1) = dilate!(erode!(img, region, dist=dist), region, dist=dist)

"""
`imgc = closing(img, [region])` performs the `closing` morphology operation, equivalent to `erode(dilate(img))`.
`region` allows you to control the dimensions over which this operation is performed.
"""
closing(img::AbstractArray, region=coords_spatial(img); dist=1) = closing!(copy(img), region, dist=dist)
closing!(img::AbstractArray, region=coords_spatial(img); dist=1) = erode!(dilate!(img, region, dist=dist), region, dist=dist)

"""
`imgth = tophat(img, [region])` performs `top hat` of an image,
which is defined as the image minus its morphological opening.
`region` allows you to control the dimensions over which this operation is performed.
"""
tophat(img::AbstractArray, region=coords_spatial(img); dist=1) = img - opening(img, region, dist=dist)

"""
`imgbh = bothat(img, [region])` performs `bottom hat` of an image,
which is defined as its morphological closing minus the original image.
`region` allows you to control the dimensions over which this operation is performed.
"""
bothat(img::AbstractArray, region=coords_spatial(img); dist=1) = closing(img, region, dist=dist) - img

"""
`imgmg = morphogradient(img, [region])` returns morphological gradient of the image,
which is the difference between the dilation and the erosion of a given image.
`region` allows you to control the dimensions over which this operation is performed.
"""
morphogradient(img::AbstractArray, region=coords_spatial(img); dist=1) = dilate(img, region, dist=dist) - erode(img, region, dist=dist)

"""
`imgml = morpholaplace(img, [region])` performs `Morphological Laplacian` of an image,
which is defined as the arithmetic difference between the internal and the external gradient.
`region` allows you to control the dimensions over which this operation is performed.
"""
morpholaplace(img::AbstractArray, region=coords_spatial(img); dist=1) = dilate(img, region, dist=dist) + erode(img, region, dist=dist) - 2img
