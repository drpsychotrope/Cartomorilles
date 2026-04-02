python -c "
from grid_builder import GridBuilder
from data_loader import DataLoader
import config
config.CELL_SIZE = 50.0
dl = DataLoader()
dem = dl.load_dem(None)
geo = dl.load_geology(None)
g = GridBuilder()
g.compute_terrain(dem)
g.score_geology(geo)
ix = int((906000 - g.x0) / g.cell_size)
iy = int((g.y0 + g.ny * g.cell_size - 6453000) / g.cell_size)
has_mask = hasattr(g, 'geology_elim_mask')
mask_val = g.geology_elim_mask[iy, ix] if has_mask else 'N/A'
geo_score = g.scores.get('geology')
gs = geo_score[iy, ix] if geo_score is not None else 'N/A'
print(f'Cell ({ix},{iy}): alt={g.altitude[iy,ix]:.0f}m  geo_elim={mask_val}  geo_score={gs}')
"