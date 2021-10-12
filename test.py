# python imports
import os
import glob
# external imports
import torch
import numpy as np
import SimpleITK as sitk
# internal imports
from Model import losses
from Model.config import args
from Model.datagenerators import Dataset
from Model.model import U_Network, SpatialTransformer


def make_dirs():
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    #print(dice_lst)
    return np.mean(dice_lst)


# @torchsnooper.snoop()
def test():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(args.checkpoint_path)

    f_img = sitk.ReadImage('/storage/caoxiaoling/LPBA/S01/brain.nii.gz')
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    # set up atlas tensor
    input_fixed = torch.from_numpy(input_fixed).to(device).float()

    # Test file and anatomical labels we want to evaluate
    test_file_lst = glob.glob(os.path.join('/storage/caoxiaoling/LPBA/test_2/*/'))
    print("The number of test data: ", len(test_file_lst))

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 64, 128]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [ 64, 32, 16, 16, 16, 16]
    # Set up model
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    check=torch.load(args.checkpoint_path,map_location='cpu')
    UNet.load_state_dict(check)
    
    STN_img = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    UNet.eval()
    STN_img.eval()
    STN_label.eval()

    DSC = []
    nj_loss_fn = losses.NJ_loss

    nj = []

    # fixed图像对应的label
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/storage/caoxiaoling/LPBA/S01/label_2.nii.gz')))
    input_hinge_2 = sitk.GetArrayFromImage(sitk.ReadImage('/storage/caoxiaoling/LPBA/S01/surf_2.nii.gz'))[np.newaxis, np.newaxis, ...]
    input_hinge_2 = torch.from_numpy(input_hinge_2).to(device).float()
    for file1 in test_file_lst:
        print(file1)
        # 读入moving图像
        input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file1+'/brain.nii.gz'))[np.newaxis, np.newaxis, ...]
        input_moving = torch.from_numpy(input_moving).to(device).float()
        # 读入moving图像对应的label
        #label_file = glob.glob(os.path.join('))
        input_label = sitk.GetArrayFromImage(sitk.ReadImage(file1+'/label_2.nii.gz'))[np.newaxis, np.newaxis, ...]
        input_label = torch.from_numpy(input_label).to(device).float()
        input_hinge = sitk.GetArrayFromImage(sitk.ReadImage(file1+'/surf_2.nii.gz'))[np.newaxis, np.newaxis, ...]
        input_hinge = torch.from_numpy(input_hinge).to(device).float()

        # 获得配准后的图像和label
        pred_flow = UNet(input_moving, input_hinge,input_fixed,input_hinge_2)
        pred_img = STN_img(input_moving, pred_flow)
        pred_label = STN_label(input_label, pred_flow)
        nj_loss=nj_loss_fn(pred_flow).item()
        print("nj: ", nj_loss)
        nj.append(nj_loss)

        # 计算DSC
        dice = compute_label_dice(fixed_label, pred_label[0, 0, ...].cpu().detach().numpy())
        print("dice: ", dice)
        DSC.append(dice)

        if 'S40' in file1:
            save_image(pred_img, f_img, "40_warped.nii.gz")
            save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, "40_flow.nii.gz")
            save_image(pred_label, f_img, "40_label.nii.gz")
        del pred_flow, pred_img, pred_label,nj_loss

    print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC)) 

    print("mean(nj): ", np.mean(nj), "   std(DSC): ", np.std(nj))    


if __name__ == "__main__":
    test()
