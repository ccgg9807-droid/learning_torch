import torch
import time
def train(net,train_data,test_data,num_epoch,optimizer,criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("正在{}上进行训练".format(device))
    net = net.to(device)

    #计时开始
    start = time.time()
    for epoch in range(num_epoch):
        #训练损失和训练准确率
        train_loss = 0 
        train_acc = 0

        #开始训练
        net.train()
        train_start = time.time()

        #注意 im = [batch_size,channel,height,width]    label = [batch_size]
        #len(train_data) = 总样本数 / batch_size 向上取整 = 循环进行的次数
        for im,label in train_data:
            #GPU加速
            im,label = im.to(device),label.to(device)

            #forward
            #[64,3,32,32] -> net -> [64,10]
            out = net(im)
            loss = criterion(out,label)

            #更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * im.shape[0]

            _,pred = out.max(1)
            train_acc += (pred == label).sum().item()
        
        train_epoch_loss = train_loss / len(train_data.dataset)
        train_epoch_acc = train_acc / len(train_data.dataset)
        train_end =time.time()
        train_time =train_end - train_start 

        test_loss = 0
        test_acc = 0
        net.eval()
        eval_start = time.time()

        for im,label in test_data:
            im,label = im.to(device),label.to(device)

            #
            out = net(im)
            loss = criterion(out,label)


            test_loss += loss.item() * im.shape[0]

            _,pred = out.max(1)
            test_acc += (pred == label ).sum().item()
        test_epoch_loss = test_loss / len(test_data.dataset)
        test_epoch_acc = test_acc / len(test_data.dataset) 
        eval_end =time.time()
        test_time = eval_end - eval_start

        print("epoch: {} train loss = {:.6f} train acc = {:.6f} train time ={:.6f} test loss = {:.6f} test acc ={:.6f} test time ={:.6f}".format(epoch,train_epoch_loss,train_epoch_acc,train_time,test_epoch_loss,test_epoch_acc,test_time))

            








