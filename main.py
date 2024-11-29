import h5py
import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.datasets import LazyHDF5Dataset, LazyHDF5Dataset_windowed
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep

def convert_to_windows(data, model):
    windows = []; w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size: w = data[i-w_size:i]
        else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
    return torch.stack(windows)

def load_dataset(dataset):
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        
        if dataset == 'SMD': file = 'machine-1-1_' + file
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        if dataset == 'UCR': file = '136_' + file
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
        if 'IMAD-DS' in dataset:
            dataset_folder = os.path.join(output_folder, dataset)
            h5_path = os.path.join(dataset_folder, 'dataset.h5')
            # WINDOW_SIZE = 1600 # 1600=100 ms -> this way different segments windows are processed separately (inpuit is composed of stakced windows of 100 ms)
            # BATCH_SIZE = 1024
            train_dataset = LazyHDF5Dataset(h5_path, dataset_name='train', n_win_per_chunk=N_WIN_PER_CHUNK)
            test_dataset = LazyHDF5Dataset(h5_path, dataset_name='test', n_win_per_chunk=N_WIN_PER_CHUNK)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            loader.append(train_loader)
            loader.append(test_loader)
            loader.append(test_dataset.labels)
            break		
        else:
            loader.append(np.load(os.path.join(folder, f'{file}.npy'), allow_pickle=True))

    # loader = [i[:, debug:debug+1] for i in loader]
    if args.less: loader[0] = cut_array(0.2, loader[0])
    if 'IMAD-DS' not in dataset:
        train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
        test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1; accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
    l = nn.MSELoss(reduction = 'mean' if training else 'none')
    feats = dataO.shape[1]
    if 'DAGMM' in model.name:
        l = nn.MSELoss(reduction = 'none')
        compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
        n = epoch + 1; w_size = model.n_window
        l1s = []; l2s = []
        if training:
            for d in data:
                _, x_hat, z, gamma = model(d)
                l1, l2 = l(x_hat, d), l(gamma, d)
                l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1) + torch.mean(l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            ae1s = []
            for d in data: 
                _, x_hat, _, _ = model(d)
                ae1s.append(x_hat)
            ae1s = torch.stack(ae1s)
            y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    if 'Attention' in model.name:
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1; w_size = model.n_window
        l1s = []; res = []
        if training:
            for d in data:
                ae, ats = model(d)
                # res.append(torch.mean(ats, axis=0).view(-1))
                l1 = l(ae, d)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            ae1s, y_pred = [], []
            for d in data: 
                ae1 = model(d)
                y_pred.append(ae1[-1])
                ae1s.append(ae1)
            ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
            loss = torch.mean(l(ae1s, data), axis=1)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'OmniAnomaly' in model.name:
        if training:
            mses, klds = [], []
            for i, d in enumerate(data):
                y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD
                mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            y_preds = []
            for i, d in enumerate(data):
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().numpy(), y_pred.detach().numpy()
    elif 'USAD' in model.name:
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1; w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d in data:
                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data: 
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1; w_size = model.n_window
        l1s = []
        if training:
            for i, d in enumerate(data):
                if 'MTAD_GAT' in model.name: 
                    x, h = model(d, h if i else None)
                else:
                    x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            xs = []
            for d in data: 
                if 'MTAD_GAT' in model.name: 
                    x, h = model(d, None)
                else:
                    x = model(d)
                xs.append(x)
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = l(xs, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'GAN' in model.name:
        l = nn.MSELoss(reduction = 'none')
        bcel = nn.BCELoss(reduction = 'mean')
        msel = nn.MSELoss(reduction = 'mean')
        real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
        real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
        n = epoch + 1; w_size = model.n_window
        mses, gls, dls = [], [], []
        if training:
            for d in data:
                # training discriminator
                model.discriminator.zero_grad()
                _, real, fake = model(d)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                model.generator.zero_grad()
                optimizer.step()
                # training generator
                z, _, fake = model(d)
                mse = msel(z, d) 
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
                # tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
        else:
            outputs = []
            for d in data: 
                z, _, _ = model(d)
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction = 'none')
        data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size = bs)
        n = epoch + 1; w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    else:
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            return loss.detach().numpy(), y_pred.detach().numpy()

if __name__ == '__main__':
    N_WIN_PER_CHUNK = 256 #number of windows to read simultaneously from memory
    BATCH_SIZE = 1 #batch size

    train_loader, test_loader, labels = load_dataset(args.dataset)
    if args.model in ['MERLIN']:
        eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

    ## Prepare data
    if 'IMAD-DS' not in args.dataset:
        trainD, testD = next(iter(train_loader)), next(iter(test_loader))
        # trainD, testD = torch.tensor(loader[0]), torch.tensor(loader[1])
        trainO, testO = trainD, testD
    if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name:
        if 'IMAD-DS' not in args.dataset:
            trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
        else:

            h5file_path = os.path.join(output_folder, args.dataset, 'dataset_windows.h5')

            if os.path.exists(h5file_path):
                print(f"File exists: {h5file_path}. Aborted creation")
            else:
                print(f"Creation of file {h5file_path}")

                WINDOW_SIZE = 1600 # 1600=100 ms -> this way different segments windows are processed separately (inpuit is composed of stakced windows of 100 ms)
                # Create HDF5 file
                with h5py.File(h5file_path, 'w') as h5file:
                    BATCH_SIZE_SAMPLES = BATCH_SIZE*N_WIN_PER_CHUNK*WINDOW_SIZE # length of the whole batch in samples
                    CHUNK_LOADED_LEN = N_WIN_PER_CHUNK*WINDOW_SIZE #length of a single chunk loaded from data loader (consists in n window of typically 1600 samples size)
                    # Determine the shape of the converted data
                    # sample_trainD = convert_to_windows(trainD, model)
                    # sample_testD = convert_to_windows(testD, model)
                    
                    # Create datasets for train and test data
                    timesteps_per_iter, window_size, n_channels = WINDOW_SIZE*N_WIN_PER_CHUNK, model.n_window, labels.shape[-1]
                    train_shape = (len(train_loader)*CHUNK_LOADED_LEN, window_size, n_channels)
                    test_shape =  (len(test_loader)*CHUNK_LOADED_LEN,  window_size, n_channels)
                    
                    train_dataset = h5file.create_dataset('train', shape=train_shape, chunks=True, dtype=np.float64)
                    test_dataset = h5file.create_dataset('test', shape=test_shape, chunks=True, dtype=np.float64)
                    labels_dataset = h5file.create_dataset('labels', shape=labels.shape, chunks=True, dtype=np.float64)
                    labels_dataset[:] = labels
                    
                    # -------------------------------------------------------- TRAIN  ---------------------------------------------------------
                    # start_time = time()
                    # # Process and store train data in chunks

                    for i, batch in tqdm(enumerate(train_loader), desc='IMAD-DS train set conversion to windows', total=len(train_loader)):
                        batch_j = torch.zeros((CHUNK_LOADED_LEN, window_size, n_channels))
                        iter_i = torch.zeros((BATCH_SIZE*CHUNK_LOADED_LEN, window_size, n_channels))
                        # data_loading_time = time()
                        # print(f"Data Loading Time: {data_loading_time - start_time:.4f} seconds")
                        # conversion_start = time()
                        for j in range(batch.shape[0]): # batch size
                            for k in range(batch.shape[1] // WINDOW_SIZE):
                                sample_window = batch[j, k * WINDOW_SIZE:(k + 1) * WINDOW_SIZE,:].squeeze(0)
                                
                                trainD = convert_to_windows(sample_window, model)

                                start_idx = k*WINDOW_SIZE
                                end_idx = start_idx + WINDOW_SIZE
                                batch_j[start_idx:end_idx, :, :] = trainD  # Convert torch tensor to numpy

                            start_idx = j*CHUNK_LOADED_LEN
                            end_idx = start_idx + CHUNK_LOADED_LEN
                            iter_i[start_idx:end_idx, :, :] = batch_j

                        # conversion_time = time()
                        # print(f"Conversion Time: {conversion_time - conversion_start:.4f} seconds")

                        start_idx = i*BATCH_SIZE_SAMPLES
                        delta = min(BATCH_SIZE_SAMPLES, batch.shape[1]) # account for shorter batches
                        end_idx = start_idx + delta
                        # data_storage_start = time()
                        train_dataset[start_idx:end_idx, :, :] = iter_i[:delta,:,:]  # Convert torch tensor to numpy
                        # data_storage_time = time()
                        # print(f"Data Storage Time: {data_storage_time - data_storage_start:.4f} seconds")

                    # -------------------------------------------------------- TEST ---------------------------------------------------------
                    for i, batch in tqdm(enumerate(test_loader), desc='IMAD-DS test set conversion to windows', total=len(test_loader)):
                        batch_j = torch.zeros((CHUNK_LOADED_LEN, window_size, n_channels))
                        iter_i = torch.zeros((BATCH_SIZE*CHUNK_LOADED_LEN, window_size, n_channels))

                        for j in range(batch.shape[0]): # batch size
                            for k in range(batch.shape[1] // WINDOW_SIZE):
                                sample_window = batch[j, k * WINDOW_SIZE:(k + 1) * WINDOW_SIZE,:].squeeze(0)
                                
                                testD = convert_to_windows(sample_window, model)

                                start_idx = k*WINDOW_SIZE
                                end_idx = start_idx + WINDOW_SIZE
                                batch_j[start_idx:end_idx, :, :] = testD  # Convert torch tensor to numpy

                            start_idx = j*CHUNK_LOADED_LEN
                            end_idx = start_idx + CHUNK_LOADED_LEN
                            iter_i[start_idx:end_idx, :, :] = batch_j

                        start_idx = i*BATCH_SIZE_SAMPLES
                        delta = min(BATCH_SIZE_SAMPLES, batch.shape[1]) # account for shorter batches
                        end_idx = start_idx + delta
                        test_dataset[start_idx:end_idx, :, :] = iter_i[:delta,:,:]  # Convert torch tensor to numpy
   
            # LOAD WINDOWED DATASET again in chunks
            # WINDOW_SIZE = 1600 # 1600=100 ms -> this way different segments windows are processed separately (inpuit is composed of stakced windows of 100 ms)
            # BATCH_SIZE = 1024

            train_dataset = LazyHDF5Dataset_windowed(h5file_path, dataset_name='train', n_win_per_chunk=N_WIN_PER_CHUNK)
            test_dataset = LazyHDF5Dataset_windowed(h5file_path, dataset_name='test', n_win_per_chunk=N_WIN_PER_CHUNK)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            # trainD, testD = train_dataset.get_chunk()
            # trainO, testO = trainD, testD

    ### Training phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = 5; e = epoch + 1; start = time()
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
        print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    ### Testing phase
    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

    ### Plot curves
    if not args.test:
        if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0)
        plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

    # ### Scores
    # df = pd.DataFrame()
    # lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    # for i in range(loss.shape[1]):
    # 	lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
    # 	result, pred = pot_eval(lt, l, ls); preds.append(pred)
    # 	df = df.append(result, ignore_index=True)

    # List to store individual DataFrames
    df_list = []
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        result, pred = pot_eval(lt, l, ls)
        preds.append(pred)
        df_list.append(pd.DataFrame(result, index=[0]))

    # Concatenate all DataFrames in the list
    df = pd.concat(df_list, ignore_index=True)

    # preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
    # pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    print(df)
    pprint(result)
    # pprint(getresults2(df, result))
    # beep(4)
