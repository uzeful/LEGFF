% This is the main function of the paper "Local extreme map guided multi-modal brain image fusion, Frontiers in Neuroscience, 2022."
% Implemented by Yu Zhang (uzeful@163.com).

% clear history and memory
clc, clear, close all;

% tag for whether storing result
is_store_res = true;

% tag for whether showing result
is_show_res = true;
if is_show_res
    h1 = figure;
end

% dataset ID:  1. PET-MRI, 2. SPECT-MRI, 3. CT-MRI, 4. infrared and visual images, 5. multi-focus images
dataID = 1; 

% set number of images in each dataset
if dataID <= 3
    ImgNum = 10;
else
    ImgNum = 20;
end

% fuse images iteratively
for ii = 1 : ImgNum
    % image index
	index = ii;

    if dataID == 1
        img1 = imread(strcat('.\Datasets\PET-MRI\MRI\', num2str(index), '.png'));
        img2 = imread(strcat('.\Datasets\PET-MRI\PET\', num2str(index), '.png'));
        fileName = ['.\Results\LEGFF-PET-MRI-', num2str(ii), '.png'];
    elseif dataID == 2
        img1 = imread(strcat('.\Datasets\SPECT-MRI\MRI\', num2str(index), '.png'));
        img2 = imread(strcat('.\Datasets\SPECT-MRI\SPECT\', num2str(index), '.png'));
        fileName = ['.\Results\LEGFF-SPECT-MRI-', num2str(ii), '.png'];
    elseif dataID == 3
        img1 = imread(strcat('.\Datasets\CT-MRI\MRI\', num2str(index), '.png'));
        img2 = imread(strcat('.\Datasets\CT-MRI\CT\', num2str(index), '.png'));
        fileName = ['.\Results\LEGFF-CT-MRI-', num2str(ii), '.png'];
    elseif dataID == 4
        img1 = imread(strcat('.\Datasets\IV2Dataset\IR', num2str(index), '.png'));
        img2 = imread(strcat('.\Datasets\IV2Dataset\VIS', num2str(index), '.png'));
        fileName = ['.\Results\LEGFF-IV2-', num2str(ii), '.png'];
    elseif dataID == 5
        img2 = imread(strcat('.\Datasets\Lytro\lytro-', num2str(index, '%02d'), '-A.jpg'));
        img1 = imread(strcat('.\Datasets\Lytro\lytro-', num2str(index, '%02d'), '-B.jpg'));
        fileName = ['.\Results\LEGFF-Lytro-lytro-', num2str(index, '%02d'), '.png'];
    end

    
    % transform color images from RGB color space to YCbCr color space if there exists color input image
    is_color = 0;
    if size(img2,3) == 3
        is_color = 1;
        LAB2 = rgb2ycbcr(img2);

        if size(img1,3) == 1
            img1 = cat(3, img1, img1, img1);
        end
        LAB1 = rgb2ycbcr(img1);

        img1 = LAB1(:,:,1);
        img2 = LAB2(:,:,1);
    end


    
    % filter images iteratively and extract multiple scales of feature maps
    bgImg1 = img1;
    bgImg2 = img2;
    Fuse_Feat = 0;
    BFeat = {};
    DFeat = {};
    scale = 5;
    for jj = 1 : scale
        % filter images
        r = jj * 2 + 1;
        [feat1, bgImg1] = imleguidedfilter(uint8(bgImg1), r);
        [feat2, bgImg2] = imleguidedfilter(uint8(bgImg2), r);
        
        % extract bright and dark feature maps    
        BFeat1{jj} = max(feat1, 0);
        BFeat2{jj} = max(feat2, 0);
        
        DFeat1{jj} = min(feat1, 0);
        DFeat2{jj} = min(feat2, 0);
        
        
        % initially fuse bright and dark feature maps, respectively, by the elementwise maximum fusion rule
        BFeat{jj} = max(BFeat1{jj}, BFeat2{jj});
        DFeat{jj} = min(DFeat1{jj}, DFeat2{jj});
        
        % compute the entropy of BFeat{jj} and that of DFeat{jj} 
        EnB(jj) = entropy(uint8(BFeat{jj}));
        EnD(jj) = entropy(uint8(-DFeat{jj}));
    end
    % computing information-amount related weights
    wEnB = (EnB + eps) / (min(EnB) + eps);
    wEnD = (EnD + eps) / (min(EnD) + eps);

    
    % fuse bright and dark feature maps with the adaptive weights
    FBFeat = 0;
    FDFeat = 0;
    for jj = 1 : scale
        FBFeat = FBFeat + wEnB(jj) * BFeat{jj};
        FDFeat = FDFeat + wEnD(jj) * DFeat{jj};
    end
    
    % fuse base images
    if dataID ~= 4
        FBase = max(bgImg1, bgImg2);
    else
        % fuse base images of the infrared and visual images by the following method
        meanBaseIR = mean(bgImg1(:));
        meanBaseIV = mean(bgImg2(:));
        diffBaseIR = bgImg1 - meanBaseIR;
        diffBaseIV = bgImg2 - meanBaseIV;
        FBase = max(meanBaseIR, meanBaseIV) + max(max(diffBaseIR, diffBaseIV), 0) + min(min(diffBaseIR, diffBaseIV), 0);      % setting in most cases
    end
    
    
    % transform the fusion image from YCbYr color space back to RGB color space
    if is_color
        LAB = LAB1;
        A1 = double(LAB1(:,:,2)); B1 = double(LAB1(:,:,3));
        A2 = double(LAB2(:,:,2)); B2 = double(LAB2(:,:,3));
        Aup = A1 .* abs(A1-127.5) + A2 .* abs(A2-127.5);
        Adw = abs(A1-127.5) + abs(A2-127.5);
        FA = Aup ./ Adw;

        Bup = B1 .* abs(B1-127.5) + B2 .* abs(B2-127.5);
        Bdw = abs(B1-127.5) + abs(B2-127.5);
        FB = Bup ./ Bdw;

        LAB(:,:,1) = uint8(FBFeat + FDFeat + FBase);
        LAB(:,:,2) = uint8(FA);
        LAB(:,:,3) = uint8(FB);
        result = ycbcr2rgb(LAB);
    else
        result = uint8(FBFeat + FDFeat + FBase);
    end

    if is_store_res
        imwrite(uint8(result), fileName)
    end
    
    if is_show_res
        figure(h1), imshow(uint8([result]))
    end
 end