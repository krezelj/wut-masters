
Position = tuple[int, int]
Positions = list[Position]


def get_neighbor_diffs(i: int, j: int, shape: tuple) -> Positions:
        neighbor_diffs = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                position = (i + di, j + dj)
                if position[0] < 0 or position[0] >= shape[0]:
                    continue
                if position[1] < 0 or position[1] >= shape[1]:
                    continue
                
                neighbor_diffs.append((di, dj))
        return neighbor_diffs


def get_neighbors(i: int, j: int, shape: tuple) -> Positions:
    diffs = get_neighbor_diffs(i, j, shape)
    neighbors = []
    for di, dj in diffs:
          neighbors.append((i + di, j + dj))
    return neighbors


def is_in_limits(i: int, j: int, shape: tuple) -> bool:
     return i >= 0 and i < shape[0] and j >= 0 and j < shape[1]