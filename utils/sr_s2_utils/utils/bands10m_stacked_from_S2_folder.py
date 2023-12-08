import os
import rasterio

def extract_10mbands_from_S2_folder(data_folder):
    """
    Extracts the Red (B04), Green (B03), Blue (B02), and Near-Infrared (B08) bands from a given Sentinel-2 data folder, 
    stacks them into a single multi-band TIFF file, and saves it to the data folder.

    The function assumes the data folder follows the Sentinel-2 SAFE format where image data is located within a subdirectory 
    named 'IMG_DATA/R10m'. It writes the output stacked TIFF file named 'stacked_10m.tif' into the provided data folder.
    
    Parameters:
    data_folder : str
        The path to the Sentinel-2 data folder. This folder should follow the structure of the '.SAFE' format.
        Default is set to an example Sentinel-2 product directory.

    Returns:
    None

    Outputs:
    A TIFF file named 'stacked_10m.tif' written in the `data_folder`. This file contains the stacked RGB and NIR bands
    with metadata (geotransform and projection) based on the Red band.

    Raises:
    FileNotFoundError: If any of the expected band files (B02, B03, B04, B08) are not found within the data folder.
    """
    
    
    # get location of image data
    for dirpath, dirnames, _ in os.walk(data_folder):
        if "IMG_DATA" in dirnames:
            folder_path = os.path.join(dirpath, "IMG_DATA")
    folder_path = os.path.join(folder_path,"R10m")
    file_paths = os.listdir(folder_path)
    image_files = {"R":os.path.join(folder_path,[file for file in file_paths if "B04" in file][0]),
                   "G":os.path.join(folder_path,[file for file in file_paths if "B03" in file][0]),
                   "B":os.path.join(folder_path,[file for file in file_paths if "B02" in file][0]),
                   "NIR":os.path.join(folder_path,[file for file in file_paths if "B08" in file][0])}
    
    
    # READ FILES
    # read and store metadata/geotrans
    with rasterio.open(image_files[list(image_files.keys())[0]]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count = len(image_files)) #GTiff
    meta.update(driver = 'GTiff')
    meta.update(dtype = rasterio.uint16)

    
    # Read each layer and stack them
    with rasterio.open(data_folder+'/stacked_10m.tif', 'w', **meta) as dst:
        for ix, (band_label, file_path) in enumerate(image_files.items(), start=1):
            print("writing band",band_label,"...")
            with rasterio.open(file_path) as src1:
                dst.write_band(ix, src1.read(1))
    dst.close()
    print("Stacked image succesfully written to:",data_folder+'stacked_10m.tif')
    
