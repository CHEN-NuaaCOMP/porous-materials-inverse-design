# Data layout

Raw XCT-derived pore sets and stress-strain labels are distributed separately.
Place aligned files under `data/full/` as described in the repository README.

Each pore-cloud CSV must provide the fields accepted by
`src.data.dataset.read_pores`. Each response CSV must provide `Strain` and
`Stress`. Sample filenames are unique identifiers; numeric prefixes do not
define parent specimens or correlated groups.

