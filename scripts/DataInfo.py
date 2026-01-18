import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.basemap import Basemap


# Load the NetCDF dataset
df = xr.open_dataset('../data/raw/WWLLN_sd_td_2005.nc/WWLLN_sd_td_2005.nc')
# print("Dataset Information:")
# print(df)
# print("\n")

# Extract latitude and longitude values
lat_values = df['lat'].values
lon_values = df['lon'].values

# Extract stroke density for May (nmon = 5, index 4 since 0-indexed)
stroke_density_may = df['stroke_density'][:, :, 0].T  # Transpose to match lat/lon grid

# Create a meshgrid for plotting
lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

# Verify shapes match
if lon_grid.shape != stroke_density_may.shape:
    raise ValueError(f'Incompatible shapes: lon_grid {lon_grid.shape} and stroke_density_may {stroke_density_may.shape}')

# Create flat map focused on Nepal
plt.figure(figsize=(14, 10))

# Create a Basemap instance for Nepal region
m = Basemap(projection='merc', llcrnrlat=26.3, urcrnrlat=30.5, 
            llcrnrlon=80.0, urcrnrlon=88.2, resolution='i')
m.drawcoastlines(linewidth=0.8)
m.drawcountries(linewidth=2.0, color='black')
m.drawmapboundary(fill_color='lightblue')

# Convert longitude and latitude to map projection coordinates
x, y = m(lon_grid, lat_grid)
# print x, y some datas
print('x shape:', x.shape)
print('y shape:', y.shape)


# plot stroke density data
contour = m.contourf(x, y, stroke_density_may, cmap='hot', levels=20)

plt.colorbar(contour, label='Stroke Density (May 2005)', shrink=0.8)
plt.title('Lightning Stroke Density - Nepal (May 2005)', fontsize=14, fontweight='bold')
plt.xlabel('Longitude (degrees)', fontsize=12)
plt.ylabel('Latitude (degrees)', fontsize=12)
plt.tight_layout()
plt.show()