from torch.nn.functional import normalize

class NormalizeScale:
    r"""Centers and normalizes node positions to (-1, 1)."""

    def __call__(self, image, occupancy, distance_field, aabb):
        maxs = image.pos.max(dim=-2, keepdim=True).values
        mins = image.pos.min(dim=-2, keepdim=True).values
        mid = (maxs + mins) / 2

        image.pos -= mid
        occupancy.pos -= mid
        aabb.min -= mid
        aabb.max -= mid

        scale = (1 / image.pos.abs().max()) * 0.999999
        image.pos *= scale
        occupancy.pos *= scale
        aabb.min *= scale
        aabb.max *= scale

        image.mid_offset = mid
        image.scale = scale
        occupancy.mid_offset = mid
        occupancy.scale = scale

        distance_field.pos = normalize(distance_field.pos, dim=1)
        distance_field.distance *= scale

        return image, occupancy, distance_field, aabb
