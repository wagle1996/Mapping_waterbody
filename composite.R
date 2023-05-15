library(raster)
library(rgdal)
library(rgeos)

#Set working directory
workingdir<-"H:\\Shared drives\\BU_CSCI1430_Image_Segmentation\\Prepared_datasets\\water_mask\\info"
setwd(workingdir)

#load raster mask files
#fn <- system.file("H:\\Shared drives\\BU_CSCI1430_Image_Segmentation\\Prepared_datasets\\water_mask\\info\\redberry.tif", package="raster")
redberry<-raster("redberry_m1.tif")
pad<-raster("pad_wa_mask.tif")

#load grid mask shapefile
GRIDred<-shapefile("redberry.shp") 
GRIDpad<-shapefile("pad.shp")

# Looking the data's 6th column which is Field
head(GRIDred@data[,6])
length(GRIDred)
head(GRIDpad@data[,6])
length(GRIDpad)

#Etracting indivual polygons of shapefile
#GRIDpad[1,]
#filename
#paste0(GRIDpad@data[i,6],".tif")

#Set output directory
outdir<-"H:\\Shared drives\\BU_CSCI1430_Image_Segmentation\\Prepared_datasets\\split_mask1"

#Loop through indivual polygon and crop and save the filename according to field
for (i in 1:length(GRIDpad)){
  cropped<-crop(redberry,GRIDpad[i,])
  filename=paste0(GRIDpad@data[i,6],".tif")
  writeRaster(cropped,filename = file.path(outdir, filename), overwrite=TRUE)
}
for (i in 1:length(GRIDred)){
  cropped<-crop(pad,GRIDred[i,])
  filename=paste0(GRIDred@data[i,6],".tif")
  writeRaster(cropped,filename = file.path(outdir, filename), overwrite=TRUE)
}


