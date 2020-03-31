function [thresh_Image, dilated_Image]=pre_processing(image_Container)
%Image Reszing
resized_Image=imresize(image_Container,[64 64]);
%Conversion from RGB to HSV
hsv_Image=rgb2hsv(resized_Image);

%Background Subtraction
thresh_Value=graythresh(hsv_Image);
thresh_Image=imbinarize(hsv_Image, thresh_Value);


%Conversion of RGB image to gray image
gray_Image=rgb2gray(resized_Image);
subtract_Image=imsubtract(gray_Image,thresh_Value);

imshow(subtract_Image);
SE  = strel('Disk',1,4);
morphologicalGradient = imsubtract(imdilate(gray_Image, SE),imerode(gray_Image, SE));
mask = imbinarize(morphologicalGradient,0.03);
SE  = strel('Disk',3,4);
mask = imclose(mask, SE);
mask = imfill(mask,'holes');
mask = bwareafilt(mask,1);
notMask = ~mask;
mask = mask | bwpropfilt(notMask,'Area',[-Inf, 5000 - eps(5000)]);
%showMaskAsOverlay(0.5,mask,'r');
gray_Image = rgb2gray(resized_Image);
gray_Image(~mask) = 255;
imshow(gray_Image)

%Image Mean Filtering
[m,n]=size(gray_Image);
filtered_Image=zeros(m,n);
for k=1:m
   for l=1:n                           
      rmin=max(1,k-1);
      rmax=min(m,k+1);
      cmin=max(1,l-1);
      cmax=min(n,l+1);
      temp=gray_Image(rmin:rmax, cmin:cmax);
      filtered_Image(k,l)=mean(temp(:));
                
    end
end
        
filtered_Image=uint8(filtered_Image);
        
        
%Sobel Dilation
edge_Sobel=edge(filtered_Image,'sobel');
edge_Sobel=imfill(edge_Sobel,'holes');
dilated_Image=edge_Sobel;
%imshow(dilated_Image);
end
