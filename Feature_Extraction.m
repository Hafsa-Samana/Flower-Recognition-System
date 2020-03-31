%Load dataset
trainingData=imageSet('flowers\Training','recursive');
for i=1:5 %5 Folders
    for j=1:769 %769 images in each Folder
        image_Container=read(trainingData(i),j);  %Reading each image and storing it
        [thresh_Image, dilated_Image]=pre_processing(image_Container);
        %imshow(image_container);
             
        
    end
end

