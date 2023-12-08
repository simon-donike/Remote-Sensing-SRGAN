import rasterio
from einops import rearrange
from rasterio.transform import Affine
import numpy as np
import torch
from tqdm import tqdm
import os

# local imports


class windowed_SR_and_saving():
    
    def __init__(self, folder_path, window_size=(128, 128),keep_lr_stack=True):
        """
        Class that performs windowed super-resolution on a Sentinel-2 image and saves the result. Steps:
        - Copies the 10m and 20m bands to new tiff files in the input directory.
        - 10m and 20m bands can be called separately and preformed with different models.
        - SR Results are saved with an averaged overlap and georeferenced in the input folder.

        Inputs:
            - folder_path: path to folder containing S2 SAFE data format
            - window_size: window size of the LR image
            - keep_lr_stack: decide wether to delete the LR stack after SR is done

        Outputs:
            - None

        Functions:
            - start_super_resolution: starts the super-resolution process. Takes model and band selection as inputs.
              Call this separately for 10m or 20m bands, sequentially
            - delete_LR_stack: deletes the LR stack after SR is done, call if not selected to do it automatically.

        Usage Example:
            # create instance of class
            sr_obj = windowed_SR_and_saving(folder_path,keep_lr_stack=True)
            # perform super-resolution on 20m bands
            sr_obj.start_super_resolution(band_selection="20m")
            # perform super-resolution on 10m bands
            sr_obj.start_super_resolution(band_selection="10m")
            # delete LR stack
            sr_obj.delete_LR_stack()
        """

        # General Settings
        self.folder_path = folder_path # path to folder containing S2 SAFE data format
        self.window_size = window_size # window size of the LR image
        self.factor=4 # sr factor of the model
        self.overlap=8 # number of pixels the windows overlap
        self.hist_match = False # wether we want to perform hist matching here
        self.keep_lr_stack = keep_lr_stack # decide wether to delete the LR stack after SR is done

        # define device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def create_and_save_placeholder_SR_files(self,info_dict,out_name):
        """
        Saves a georeferenced placeholder SR file in the input folder.
        """
        # create placeholder tensor in memory
        sr_tensor_placeholder = np.zeros((len(info_dict["bands"]), 
                                               info_dict["img_width"]*self.factor,
                                               info_dict["img_height"]*self.factor),dtype=info_dict["dtype"])
        
        # change geotransform to reflect smaller SR pixel size
        save_transform = Affine(
                info_dict["geo_transform"].a / self.factor, 
                info_dict["geo_transform"].b, 
                info_dict["geo_transform"].c, 
                info_dict["geo_transform"].d, 
                info_dict["geo_transform"].e / self.factor, 
                info_dict["geo_transform"].f
            )

        # create Metadata for rasterio saveing
        meta = {
        'driver': 'GTiff',
        'dtype': info_dict["dtype"],  # Ensure dtype matches your array's dtype
        'nodata': None,  # Set to your no-data value, if applicable
        'width': info_dict["img_width"]*self.factor,
        'height': info_dict["img_height"]*self.factor,
        'count': len(info_dict["bands"]),  # Number of bands; adjust if your array has multiple bands
        'crs': info_dict["crs"],  # CRS (Coordinate Reference System); set as needed
        'transform': save_transform,  # Adjust as needed
                }

        # Create and write SR placeholder to the raster file
        with rasterio.open(os.path.join(self.folder_path,out_name), 'w', **meta) as dst:
            # Assuming 'your_array' is 2D, write it to the first band
            for band in range(sr_tensor_placeholder.shape[0]):
                dst.write(sr_tensor_placeholder[band, :, :], band + 1)
        print("Saved empty placeholder SR image at: ",os.path.join(self.folder_path,out_name))
        # return file path of placeholder
        return os.path.join(self.folder_path,out_name)
        
    
    def create_window_coordinates_overlap(self,info_dict):
        """
        Creates a list of window coordinates for the input image. The windows overlap by a specified amount.
        Output type is a list of rasterio.windows.Window objects.
        """
        overlap = self.overlap
        # Calculate the number of windows in each dimension
        n_windows_x = (info_dict["img_width"] - overlap) // (self.window_size[0] - overlap)
        n_windows_y = (info_dict["img_height"] - overlap) // (self.window_size[1] - overlap)

        # Create list of batch windows coordinates
        window_coordinates = []
        for win_y in range(n_windows_y):
            for win_x in range(n_windows_x):
                # Define window to read with overlap
                window = rasterio.windows.Window(
                    win_x * (self.window_size[0] - overlap),
                    win_y * (self.window_size[1] - overlap),
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        # Check for any remaining space after the sliding window approach
        final_x = info_dict["img_width"] - self.window_size[0]
        final_y = info_dict["img_height"] - self.window_size[1]

        # Add extra windows for the edges if there's remaining space
        # Adjust the check to handle the overlap correctly
        if final_x % (self.window_size[0] - overlap) > 0:
            for win_y in range(n_windows_y):
                window = rasterio.windows.Window(
                    info_dict["img_width"] - self.window_size[0],
                    win_y * (self.window_size[1] - overlap),
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        if final_y % (self.window_size[1] - overlap) > 0:
            for win_x in range(n_windows_x):
                window = rasterio.windows.Window(
                    win_x * (self.window_size[0] - overlap),
                    info_dict["img_height"] - self.window_size[1],
                    self.window_size[0],
                    self.window_size[1]
                )
                window_coordinates.append(window)

        # Handle corner case if both x and y have remaining space
        if (final_x % (self.window_size[0] - overlap) > 0 and
                final_y % (self.window_size[1] - overlap) > 0):
            window = rasterio.windows.Window(
                info_dict["img_width"] - self.window_size[0],
                info_dict["img_height"] - self.window_size[1],
                self.window_size[0],
                self.window_size[1]
            )
            window_coordinates.append(window)

        # Return filled list of coordinates
        return window_coordinates
    
    
    def get_window(self,idx,info_dict):
        """
        Loads a window of the input image and returns it as a tensor.
        """
        # TODO: perform batched SR instead of single image per batch
        # assert number required is valid
        assert idx>=0 and idx<len(info_dict["window_coordinates"]), "idx not in range of windows"
        # get window of current idx
        current_window = info_dict["window_coordinates"][idx]
        
        # open file SRC
        with rasterio.open(info_dict["lr_path"]) as src:
            data = src.read(window=current_window)
            # select bands
            data = data[info_dict["bands"],:,:]
        
        # TRANSFORMS
        # TODO: define proper norm/transforms. Either in model or here, but not both
        # bring to range 0..1
        data = data/10000
        data = torch.from_numpy(data)
        # bring data to range -1..+1
        #data = (data*2)-1
        
        # return array of window that has been read
        return(data)
    
    def delete_LR_stack(self,info_dict):
        # delete LR stack
        os.remove(info_dict["lr_path"])
        print("Deleted stacked image at",info_dict["lr_path"])
        
    def fill_SR_overlap(self, sr, idx, info_dict):
        """
        Fills the SR placeholder image with the super-resoluted window at the correct location of the image. Performs windowed writing via rasterio.
        """

        # If input nor np.array, transform to array
        sr = sr.numpy() if isinstance(sr, torch.Tensor) else sr
        
        overlap = self.overlap

        # Get coor of idx window, create new rasterio Windoow in which it should be saved
        current_window = info_dict["window_coordinates"][idx]
        row_off, col_off = current_window.row_off * 4, current_window.col_off * 4
        sr_file_window = rasterio.windows.Window(col_off,row_off, sr.shape[-2], sr.shape[-1])

        # Get sr shape info
        num_channels, win_height, win_width = sr.shape

        # change sr data range and dtype to correspond with original
        sr = sr*10000
        sr = np.array(sr, dtype=info_dict["dtype"])
        sr = sr.astype(info_dict["dtype"])

        # save to placehodler .tiff on disk
        # Open the TIFF file in 'r+' mode (read/write mode)
        with rasterio.open(info_dict["sr_path"], 'r+') as dst:
            # Check if the number of bands in the tensor matches the TIFF file
            if dst.count != sr.shape[0]:
                #print("DST count:",dst.count," - SR bands: ",sr.shape[0])
                raise ValueError("The number of bands in the tensor does not match the TIFF file.")
            # Write each band of the tensor to the corresponding band in the raster
            for band in range(sr.shape[0]):
                dst.write(sr[band, :, :], band + 1, window=sr_file_window)

    
    
    def super_resolute_bands(self,info_dict,model=None,forward_call="forward",custom_steps=200):
        
        """
        Super-resolutes the entire image of the class using a specified or default super-resolution model.

        Parameters:
        -----------
        info_dict : dict
            A dictionary containing information about the image to super-resolute. This is generated automatically
            if the class has been initialized and called correctly (via the start_super_resolution).

        model : object, optional
            A PyTorch model instance that performs super-resolution. If not specified, 
            a default SR model using bilinear interpolation is used.

        forward_call : str, optional
            The name of the method to call on the model for performing super-resolution.
            Defaults to "forward". You can specify any custom method that your model has for
            super-resolution.

        Usage:
        ------
        sr_instance.super_resolute_bands(info_dict, model=my_model, forward_call="custom_forward")

        Notes:
        ------
        - The model provided (or the default one) should have a method corresponding to the name 
          passed in `forward_call` that takes a low-res image and returns a super-resoluted version.

        Returns:
        --------
        None : Saves the super-resoluted image in the SR placeholder on the disk via windowed writing.

        """
        # Get interpolation model if model not specified
        if model==None:
            # Create SR mock-up
            class sr_model():
                def __init__(self,custom_steps=200):
                    pass
                def forward(self,lr,custom_steps=200):
                    sr = torch.nn.functional.interpolate(lr, size=(512, 512), mode='bilinear', align_corners=False)
                    return(sr)
            model = sr_model()
            
        # allow custom defined forward/SR call on model
        model_sr_call = getattr(model, forward_call,custom_steps)

        # iterate over image batches
        print("Super-Resoluting ...")
        for idx in tqdm(range(len(info_dict["window_coordinates"])),ascii=False):
            # get image from S2 image
            im = self.get_window(idx,info_dict)
            # batch image
            im = im.unsqueeze(0)
            # turn into wanted dtype (double)
            im = im.float()
            # send to device
            im = im.to(self.device)
            # if ddpm, prepare for dictionary input
            if forward_call == "perform_custom_inf_step":
                im = {"LR_image":im,"image":torch.rand(im.shape[0],im.shape[1],512,512)}
            
            # super-resolute image
            sr = model_sr_call(im)

            # try to move to CPu
            try:
                sr = sr.detach().cpu()
            except:
                pass
            
            # save SR into image
            self.fill_SR_overlap(sr[0],idx,info_dict)

        # when done, save array into same directory
        print("Finished. SR image saved at",info_dict["sr_path"])

    def start_super_resolution(self,band_selection="10m",model=None,forward_call="forward"):
        
        # assert band selection in available implemented methods
        assert band_selection in ["10m","20m"], "band_selection not in ['10m','20m']"        

        # select info dictionary:
        # 1 .Get file info from Rasterio
        # 2. Create window coordinates for selected bands
        # 3. Create and save placeholder SR file
        if band_selection=="10m":
            # if it doesnt exist, create stacked LR images
            if not os.path.exists(os.path.join(self.folder_path,"stacked_10m.tif")):
                print("Stacked 10m bands not found, creating ...")
                
                from utils.sr_s2_utils.utils.bands10m_stacked_from_S2_folder import extract_10mbands_from_S2_folder
                extract_10mbands_from_S2_folder(self.folder_path)

            # Get File information - 10m bands
            self.b10m_file_path = os.path.join(self.folder_path,"stacked_10m.tif")
            self.b10m_info = {}
            self.b10m_info["lr_path"] = self.b10m_file_path
            self.b10m_info["bands"] = [0,1,2]
            with rasterio.open(self.b10m_file_path) as src:
                self.b10m_info["img_width"], self.b10m_info["img_height"],self.b10m_info["dtype"] = src.width, src.height,src.dtypes[0]
                # Extract the affine transformation matrix
                self.b10m_info["geo_transform"] = src.transform
                # Extract the CRS
                self.b10m_info["crs"] = src.crs
            
            # call local functions: get windxw coordinates and create placeholder SR file
            self.b10m_info["window_coordinates"] = self.create_window_coordinates_overlap(self.b10m_info)
            self.b10m_info["sr_path"] = self.create_and_save_placeholder_SR_files(self.b10m_info,out_name="SR_10mbands.tif")
            info_dict = self.b10m_info
        
        
        if band_selection=="20m":
            # if it doesnt exist, create stacked LR images
            if not os.path.exists(os.path.join(self.folder_path,"stacked_20m.tif")):
                print("Stacked 20m bands not found, creating ...")
                from utils.sr_s2_utils.utils.bands20m_stacked_from_S2_folder import extract_20mbands_from_S2_folder
                extract_20mbands_from_S2_folder(self.folder_path)

            # Get File information - 20m bands 
            self.b20m_file_path = os.path.join(self.folder_path,"stacked_20m.tif")
            self.b20m_info = {}
            self.b20m_info["lr_path"] = self.b20m_file_path
            self.b20m_info["bands"] = [0,1,2,3,4,5]
            with rasterio.open(self.b20m_file_path) as src:
                self.b20m_info["img_width"], self.b20m_info["img_height"],self.b20m_info["dtype"] = src.width, src.height, src.dtypes[0]
                # Extract the affine transformation matrix
                self.b20m_info["geo_transform"] = src.transform
                # Extract the CRS
                self.b20m_info["crs"] = src.crs

            # call local functions: get windxw coordinates and create placeholder SR file
            self.b20m_info["window_coordinates"] = self.create_window_coordinates_overlap(self.b20m_info)
            self.b20m_info["sr_path"] = self.create_and_save_placeholder_SR_files(self.b20m_info,out_name="SR_20mbands.tif")
            info_dict = self.b20m_info

        # perform Super-resolution of the wanted bands
        self.super_resolute_bands(info_dict,model)

        # if wanted, delete LR stack
        if not self.keep_lr_stack:
            self.delete_LR_stack(info_dict)


