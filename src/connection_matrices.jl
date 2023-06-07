# [[file:../SimpleUpdate.org::*Connection Matrices][Connection Matrices:1]]
module ConnectionMatrices
using ..Util: connection_matrix_from_connections
export connection_matrix_dict
# Connection Matrices:1 ends here

# [[file:../SimpleUpdate.org::*Connection Matrices][Connection Matrices:2]]
M_PEPS_floretpentagon =
    [
        ((1, 1, 2), (2, 1, 3)),
        ((2, 1, 1), (3, 1, 2)),
        ((3, 1, 3), (5, 1, 2)),
        ((3, 1, 1), (4, 1, 3)),
        ((4, 1, 1), (6, 1, 2)),
        ((6, 1, 1), (7, 1, 3)),
        ((7, 1, 1), (8, 1, 2)),
        ((7, 1, 2), (9, 1, 3)),
        ((5, 1, 1), (1, 2, 4)),
        ((6, 1, 3), (1, 2, 3)),
        ((8, 1, 3), (2, 2, 2)),
        ((8, 1, 1), (1, 4, 5)),
        ((9, 1, 1), (5, 3, 3)),
        ((9, 1, 2), (1, 3, 1)),
        ((4, 1, 2), (1, 3, 6)),
    ] |> connection_matrix_from_connections

M_PESS_floretpentagon_petaltwirl =
    [
        ((3, 1, 1), (4, 1, 3), (5, 1, 1), (6, 1, 3), (1, 2, 3)),
        ((6, 1, 1), (7, 1, 3), (8, 1, 3), (1, 2, 2), (2, 2, 2)),
        ((4, 1, 1), (6, 1, 2), (7, 1, 2), (9, 1, 3), (1, 3, 6)),
        ((9, 1, 2), (1, 3, 1), (2, 3, 3), (3, 3, 3), (5, 3, 2)),
        ((7, 1, 1), (8, 1, 2), (9, 1, 1), (5, 3, 3), (1, 4, 4)),
        ((8, 1, 1), (2, 2, 1), (3, 2, 2), (4, 2, 2), (1, 4, 5)),
    ] |> connection_matrix_from_connections

M_PESS_square =
    [((1, 1, 1), (1, 2, 2), (1, 3, 4), (1, 4, 3))] |> connection_matrix_from_connections

M_PESS_kagome_3 =
    [((1, 1, 1), (2, 1, 2), (3, 1, 1)), ((1, 4, 2), (2, 2, 1), (3, 3, 2))] |>
    connection_matrix_from_connections

M_PESS_triangular =
    connection_matrix_from_connections([((1, 1, 1), (1, 2, 2), (1, 3, 3))], 4)

connection_matrix_dict = Dict(
    "PEPS_floretpentagon" => M_PEPS_floretpentagon,
    "PESS_floretpentagon_petaltwirl" => M_PESS_floretpentagon_petaltwirl,
    "PESS_square" => M_PESS_square,
    "PESS_kagome_3" => M_PESS_kagome_3,
)
# Connection Matrices:2 ends here

# [[file:../SimpleUpdate.org::*Connection Matrices][Connection Matrices:3]]
end
# Connection Matrices:3 ends here
