from os.path import join

class FLAGES(object):
    pan_size= 32
    ms_size=pan_size//4
    num_spectrum=4
    ratio=4
    stride=32
    num_workers=1
    test_batch_size=1
    train_batch_size=64
    total_epochs=1000
    steps_per_epoch=100
    version=3
    model_backup_freq=10
    backup_model=0
    satellite = 'gf1'
    method = 'transpan'
    optimizer = 'adam'
    lr =1e-4
    beta1 = 0.9
    beta2 = 0.999
    if_l1_loss=True
    pth='best'
    embed_dim=512
    patch_size=4
    depth=3
    num_heads=8

    img_path = "D:\\file_wn\\work3_competed_methods\\DataSet\\3GF1\\"
    data_path = "../dataloader/3GF1/"
    train_img_filename="train_{}_v1.h5".format(satellite)

    record_loss_file = '../pnn-results/results{}/record_loss.txt'.format(version)  # 保存训练阶段的损失值
    model_dir = '../pnn-results/results{}/model/'.format(version)
    fused_images_dir = '../pnn-results/results{}/{}/{}_{}/fused_images/'.format(version,version,version,pth)
    visual_img_save_path = '../pnn-results/results{}/visual_images/'.format(version)
    backup_model_dir = join(model_dir, 'backup_model/')

