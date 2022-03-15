import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pre_weights="./checkpoints/CRNN-1010.pth"
pretext_model=torch.load(pre_weights,map_location=torch.device('cpu'))


next_weight=r"D:\ocrcar\train_code\train_crnn\crnn_models\CRNN-1011.pth"
next_text_model=torch.load(next_weight,map_location=torch.device('cpu'))
# for k in next_text_model:
#     print(k)
next_text_model['layer1.0.weight']=pretext_model['conv1.weight']
next_text_model['layer1.0.bias']=pretext_model['conv1.bias']

next_text_model['layer2.0.weight']=pretext_model['conv2.weight']
next_text_model['layer2.0.bias']=pretext_model['conv2.bias']

next_text_model['layer3.0.weight']=pretext_model['conv3_1.weight']
next_text_model['layer3.0.bias']=pretext_model['conv3_1.bias']

next_text_model['layer3.1.weight']=pretext_model['bn3.weight']
next_text_model['layer3.1.bias']=pretext_model['bn3.bias']

next_text_model['layer3.1.running_mean']=pretext_model['bn3.running_mean']
next_text_model['layer3.1.running_var']=pretext_model['bn3.running_var']
next_text_model['layer3.1.num_batches_tracked']=pretext_model['bn3.num_batches_tracked']

next_text_model['layer3.3.weight']=pretext_model['conv3_2.weight']
next_text_model['layer3.3.bias']=pretext_model['conv3_2.bias']

next_text_model['layer4.0.weight']=pretext_model['conv4_1.weight']
next_text_model['layer4.0.bias']=pretext_model['conv4_1.bias']

next_text_model['layer4.1.weight']=pretext_model['bn4.weight']
next_text_model['layer4.1.bias']=pretext_model['bn4.bias']

next_text_model['layer4.1.running_mean']=pretext_model['bn4.running_mean']
next_text_model['layer4.1.running_var']=pretext_model['bn4.running_var']
next_text_model['layer4.1.num_batches_tracked']=pretext_model['bn4.num_batches_tracked']

next_text_model['layer4.3.weight']=pretext_model['conv4_2.weight']
next_text_model['layer4.3.bias']=pretext_model['conv4_2.bias']

next_text_model['layer5.0.weight']=pretext_model['conv5.weight']
next_text_model['layer5.0.bias']=pretext_model['conv5.bias']
next_text_model['layer5.1.weight']=pretext_model['bn5.weight']
next_text_model['layer5.1.bias']=pretext_model['bn5.bias']

next_text_model['layer5.1.running_mean']=pretext_model['bn5.running_mean']
next_text_model['layer5.1.running_var']=pretext_model['bn5.running_var']
next_text_model['layer5.1.num_batches_tracked']=pretext_model['bn5.num_batches_tracked']

next_text_model['rnn.0.rnn.weight_ih_l0']=pretext_model['rnn.0.rnn.weight_ih_l0']
next_text_model['rnn.0.rnn.weight_hh_l0']=pretext_model['rnn.0.rnn.weight_hh_l0']
next_text_model['rnn.0.rnn.bias_ih_l0']=pretext_model['rnn.0.rnn.bias_ih_l0']
next_text_model['rnn.0.rnn.bias_hh_l0']=pretext_model['rnn.0.rnn.bias_hh_l0']
next_text_model['rnn.0.rnn.weight_ih_l0_reverse']=pretext_model['rnn.0.rnn.weight_ih_l0_reverse']
next_text_model['rnn.0.rnn.weight_hh_l0_reverse']=pretext_model['rnn.0.rnn.weight_hh_l0_reverse']
next_text_model['rnn.0.rnn.bias_ih_l0_reverse']=pretext_model['rnn.0.rnn.bias_ih_l0_reverse']
next_text_model['rnn.0.rnn.bias_hh_l0_reverse']=pretext_model['rnn.0.rnn.bias_hh_l0_reverse']
next_text_model['rnn.0.embedding.weight']=pretext_model['rnn.0.embedding.weight']
next_text_model['rnn.0.embedding.bias']=pretext_model['rnn.0.embedding.bias']
next_text_model['rnn.1.rnn.weight_ih_l0']=pretext_model['rnn.1.rnn.weight_ih_l0']
next_text_model['rnn.1.rnn.weight_hh_l0']=pretext_model['rnn.1.rnn.weight_hh_l0']
next_text_model['rnn.1.rnn.bias_ih_l0']=pretext_model['rnn.1.rnn.bias_ih_l0']
next_text_model['rnn.1.rnn.bias_hh_l0']=pretext_model['rnn.1.rnn.bias_hh_l0']
next_text_model['rnn.1.rnn.weight_ih_l0_reverse']=pretext_model['rnn.1.rnn.weight_ih_l0_reverse']
next_text_model['rnn.1.rnn.weight_hh_l0_reverse']=pretext_model['rnn.1.rnn.weight_hh_l0_reverse']
next_text_model['rnn.1.rnn.bias_ih_l0_reverse']=pretext_model['rnn.1.rnn.bias_ih_l0_reverse']
next_text_model['rnn.1.rnn.bias_hh_l0_reverse']=pretext_model['rnn.1.rnn.bias_hh_l0_reverse']
next_text_model['rnn.1.embedding.weight']=pretext_model['rnn.1.embedding.weight']
next_text_model['rnn.1.embedding.bias']=pretext_model['rnn.1.embedding.bias']

torch.save(next_text_model, './checkpoints/CRNN.pth')




if __name__ == '__main__':
   pass