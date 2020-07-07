function ss = LBPsim(img1,img2);

% This code LBPsim is proposed for the evaluation of perceptual visual security assessment.
% The author is Doctor Jian Xiong (jxiong AT njupt Dot edu Dot cn).

%The source code of the function efficientLBP() can be downloaded via the link 
%(https://ww2.mathworks.cn/matlabcentral/fileexchange/49787-shift-based-lbp?s_tid=srchtitle)

%============parameter settings===============

    C1 = 0.1^10;
    nFiltSize=8;
    nFiltRadius=1;
    binsRange=(1:2^nFiltSize)-1;
    filtR=generateRadialFilterLBP(nFiltSize, nFiltRadius);
    [row,col]=size(img1);
    gamma = 0.2;
    zeta = 0.15;
    
    
%=============dct based  ========================

    f_ref =abs(dct2(img1));
    f_dst = abs(dct2(img2));
    sim_dct = (2*f_ref.*f_dst + C1)./(f_ref.*f_ref+f_dst.*f_dst + C1);
       
    [G_ref]=gradientxy(img1);
    mask2 = getmask(G_ref,gamma);
       
         
%=============lbp based similarity =========================================================
    [LBP1,biword1]   = efficientLBP(img1, 'filtR', filtR, 'isRotInv', false, 'isChanWiseRot', false);
    [LBP2,biword2]   = efficientLBP(img2, 'filtR', filtR, 'isRotInv', false, 'isChanWiseRot', false);
    sim_lbp_loc= 1-mean(xor(biword1,biword2),3);
    
    img1_d = downsample(img1,4);
    img2_d = downsample(img2,4);
    [effLBP1,binaryword1]   = efficientLBP(img1_d, 'filtR', filtR, 'isRotInv', false, 'isChanWiseRot', false);
    [effLBP2,binaryword2]   = efficientLBP(img2_d, 'filtR', filtR, 'isRotInv', false, 'isChanWiseRot', false);
    sim_lbp_loc4= 1-mean(xor(binaryword1,binaryword2),3);
    sim_lbp_loc4 = upsample(sim_lbp_loc4,row,col,4);
    hist1 = hist(single( effLBP1(:) ), binsRange);
    hist2 = hist(single( effLBP2(:) ), binsRange);
    sim_lbp4 = (2*hist1.*hist2 + C1)./(hist1.*hist1+hist2.*hist2 + C1); 
    
    img1_d = downsample(img1,16);
    img2_d = downsample(img2,16);
    [effLBP1,binaryword1]   = efficientLBP(img1_d, 'filtR', filtR, 'isRotInv', false, 'isChanWiseRot', false);
    [effLBP2,binaryword2]   = efficientLBP(img2_d, 'filtR', filtR, 'isRotInv', false, 'isChanWiseRot', false);
    sim_lbp_loc16= 1-mean(xor(binaryword1,binaryword2),3);
    sim_lbp_loc16 = upsample(sim_lbp_loc16,row,col,16);
    hist1 = hist(single( effLBP1(:) ), binsRange);
    hist2 = hist(single( effLBP2(:) ), binsRange);
    sim_lbp16 = (2*hist1.*hist2 + C1)./(hist1.*hist1+hist2.*hist2 + C1);
    
%===================== Guassian function approximated low-pass CSF model==
    [x,y] = meshgrid(0:max(size(sim_dct))-1);
    [row,col ] = size(sim_dct); 
    sigma =   zeta*(row+col);
    z = 1/(sqrt(2*pi)*sigma).*exp((-x.^2-y.^2)/sigma^2);
    z = z./z(1,1);
%     mesh(z);
    S_dct = mean2(sim_dct.*z(1:row,1:col));    
    SG = mean(sim_lbp16) * mean(sim_lbp4);  
    SL = mean2(sim_lbp_loc.*sim_lbp_loc4.*sim_lbp_loc16.*mask2);
    
    ss =  S_dct*SG*SL;
end
