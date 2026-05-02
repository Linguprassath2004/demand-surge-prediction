import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pygeohash as pgh
import matplotlib.pyplot as plt
import contextily as cx
from src.config import DATA_PROCESSED, MODEL_PATH
from src.models.predict_model import load_model, predict

def visualize_model_predictions(timestamp_to_visualize, output_path="predicted_demand_map.png"):
    # 1. Load the processed data and the trained model
    df = pd.read_csv(DATA_PROCESSED, parse_dates=['timestamp'])
    model = load_model(MODEL_PATH)

    # 2. Filter data for the requested timestamp
    df_slice = df[df['timestamp'] == timestamp_to_visualize]

    if df_slice.empty:
        print(f"No data found for timestamp: {timestamp_to_visualize}")
        return

    # 3. Define features (matching the list used in training)
    features = [col for col in df.columns if col not in ['timestamp', 'geohash6', 'demand']]

    # 4. Generate predictions
    X = df_slice[features]
    df_slice['predicted_demand'] = predict(model, X)

    # 5. Decode geohashes into latitude and longitude
    def decode_geohash(gh):
        try:
            lat, lon = pgh.decode(gh)
            return pd.Series({'latitude': lat, 'longitude': lon})
        except:
            return pd.Series({'latitude': None, 'longitude': None})

    coords = df_slice['geohash6'].apply(decode_geohash)
    df_slice = pd.concat([df_slice.reset_index(drop=True), coords], axis=1)
    df_slice = df_slice.dropna(subset=['latitude', 'longitude'])

    # 6. Create a GeoDataFrame
    geometry = [Point(xy) for xy in zip(df_slice['longitude'], df_slice['latitude'])]
    gdf = gpd.GeoDataFrame(df_slice, crs="EPSG:4326", geometry=geometry)
    
    # Convert CRS to Web Mercator (EPSG:3857) for base map compatibility
    gdf_mercator = gdf.to_crs(epsg=3857)

    # 7. Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 10))

    gdf_mercator.plot(
        column='predicted_demand',
        cmap='YlOrRd',
        markersize=gdf_mercator['predicted_demand'] * 100,
        alpha=0.6,
        ax=ax,
        legend=True,
        legend_kwds={'label': 'Predicted Demand', 'shrink': 0.5}
    )

    # Add a contextual basemap
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

    ax.set_title(f"Predicted Demand Surge Heatmap for {timestamp_to_visualize}", fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Map saved to {output_path}")

if __name__ == "__main__":
    # Example usage (update timestamp based on your dataset range):
    # visualize_model_predictions('2023-11-01 08:00:00')
    pass