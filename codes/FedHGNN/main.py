from client import Client
from server import Server
import os
import dgl
from time import time
import sys

from ourparse import *
from scipy.sparse import lil_matrix
from model import model
from rec_dataset import *
import torch.utils.data as dataloader
import copy
import random
from util import *
print("use:", torch.device(args.device))


'''seed'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20211111)
'''seed end'''

'''log'''
file = os.path.basename(sys.argv[0])[0:-3]+"_"+str(time())
print_log(args.log_dir+file)
'log end'

def train_test_split(p_vs_a):
    train_id = []
    train_fed_id = []
    test_id = []
    test_negative_id = []
    p_vs_a_ = copy.deepcopy(p_vs_a)#
    p_vs_a_random = copy.deepcopy(p_vs_a)
    p_vs_a_random = p_vs_a_random.tolil()
    p_num = p_vs_a_.shape[0]
    a_num = p_vs_a_.shape[1]
    for i in range(p_num):#each paper
        cur_a = p_vs_a_[i].nonzero()[1]
        '''p_vs_a random'''
        p_vs_a_random[i,:]=0
        sample_len = len(cur_a)
        sample_a = random.sample(list(range(p_vs_a_.shape[1])), sample_len)
        #print(sample_a)
        p_vs_a_random[i, sample_a] = 1
        # print(p_vs_a_random[i].nonzero()[1])
        '''end'''

        if(len(cur_a)==1):
            train_id.append([i, cur_a[0]])
            train_fed_id.append(list(cur_a))#
        elif(len(cur_a)!=0):
            sample_train = random.sample(list(cur_a), len(cur_a)-1)
            train_fed_id.append(sample_train)#
            for j in sample_train:
                train_id.append([i, j])
            cur_test_id =list(set(cur_a)-set(sample_train))[0]
            test_id.append([i, cur_test_id])
            p_vs_a_[i, cur_test_id] = 0

            '''p_vs_a random'''
            p_vs_a_random[i, cur_test_id] = 0#random
            '''end'''

            test_negative_pool = list(set(range(a_num))-set(cur_a))#0-10... -
            test_negative_id.append(random.sample(test_negative_pool, 99))
        else:
            train_fed_id.append([])
    #print(len(train_fed_id))
    #print(test_negative_id[2])
    return p_vs_a_, p_vs_a_random, train_fed_id, train_id, test_id, test_negative_id


dataname = args.dataset
device = args.device

meta_paths_dict = {'acm':{'user': [['pa','ap'],['pc','cp']],'item':[['ap','pa']]}, \
                   'dblp':{'user': [['pa','ap'], ['pc','cp']],'item':[['ap','pa']]}, \
                   'yelp':{'user': [['pa','ap'], ['pa','aca','caa', 'ap']],'item':[['ap', 'pa']]}, \
                   'DoubanBook':{'item':[['bu','ub'],['bg', 'gb']], 'user':[['ub','bu'],['ua','au']]}}

data_path = '../../data/ACM/ACM.mat'
data = sio.loadmat(data_path)
p_vs_f = data['PvsL']
p_vs_a = data['PvsA']#(12499, 17431)
p_vs_t = data['PvsT']
p_vs_c = data['PvsC']#(12499, 14)



'''test'''

adj = (p_vs_f, p_vs_a, p_vs_t, p_vs_c)
label_count, labels, label_author, author_label, shared_knowledge_rep = gen_shared_knowledge(adj, args.shared_num)#
'''test end'''

# We assign
# (1) KDD papers as class 0 (data mining),
# (2) SIGMOD and VLDB papers as class 1 (database),
# (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
conf_ids = [0, 1, 9, 10, 13]
label_ids = [0, 1, 2, 2, 1]

p_vs_c_filter = p_vs_c[:, conf_ids]#(12499, 5)
p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
# print(type(p_vs_c_filter.sum(1) != 0))
p_vs_f = p_vs_f[p_selected]#(4025,73)
p_vs_a = p_vs_a[p_selected]#(4025,17431)
p_vs_t = p_vs_t[p_selected]#(4025,1903)
p_vs_c = p_vs_c[p_selected]#CSC (4025, 14)



num_nodes_dict = {'paper': p_vs_a.shape[0], 'author': p_vs_a.shape[1], 'field': p_vs_f.shape[1], 'conf': p_vs_c.shape[1]}


p_vs_a, p_vs_a_random, train_fed_id, train_id, test_id, test_negative_id=train_test_split(p_vs_a)

logging.info(args)
logging.info(meta_paths_dict)


#features_user = torch.FloatTensor(p_vs_t.toarray())
features_user = np.random.normal(loc=0., scale=1., size=[p_vs_a.shape[0], args.in_dim])
features_item = np.random.normal(loc=0., scale=1., size=[p_vs_a.shape[1], args.in_dim])
features = (features_user, features_item)
#print(features.shape)


model_user = model(meta_paths=meta_paths_dict[dataname]['user'],
                   in_size=features_user.shape[1],
                   hidden_size=8,
                   out_size=64,
                   num_heads=args.num_heads,
                   dropout=args.dropout).to(device)

model_item = model(meta_paths=meta_paths_dict[dataname]['item'],
                   in_size=features_item.shape[1],
                   hidden_size=8,
                   out_size=64,
                   num_heads=args.num_heads,
                   dropout=args.dropout).to(device)#0.6
model = (model_user, model_item)

test_dataset = RecDataSet(test_id, test_negative_id, is_training=False)
test_dataloader = dataloader.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)



client_list = []

p_vs_a_ = []
all_edges=0
remain_edges=0
all_edges_after=0
for i, items in enumerate(train_fed_id):#each paper
    pre_edges = list(p_vs_a[i].nonzero()[1])
    all_edges+=len(pre_edges)

    client = Client(i, items, args)
    perturb_adj = client.perturb_adj(p_vs_a.todense()[i], label_author, author_label, label_count,
                                     shared_knowledge_rep, args.p1, args.p2)

    cur_edges = list(lil_matrix(perturb_adj).nonzero()[1])
    all_edges_after+=len(cur_edges)
    cur_remain_edges = len(set(pre_edges)&set(cur_edges))
    remain_edges+=cur_remain_edges
    p_vs_a_.append(perturb_adj)
    client_list.append(client.to(torch.device(args.device)))

p_vs_a_ = np.squeeze(np.array(p_vs_a_))
p_vs_a_ = lil_matrix(p_vs_a_)


print(all_edges)
print(all_edges_after)
print(remain_edges)
print(remain_edges/all_edges)

hg = dgl.heterograph({
    ('paper', 'pa', 'author'): p_vs_a_.nonzero(),
    ('author', 'ap', 'paper'): p_vs_a_.transpose().nonzero(),
    ('paper', 'pf', 'field'): p_vs_f.nonzero(),
    ('field', 'fp', 'paper'): p_vs_f.transpose().nonzero(),
    ('paper', 'pc', 'conf'): p_vs_c.nonzero(),
    ('conf', 'cp', 'paper'): p_vs_c.transpose().nonzero(),
}, num_nodes_dict = num_nodes_dict).to(device)


print(p_vs_f.shape)
print(p_vs_a.shape)
print(p_vs_t.shape)
print(p_vs_c.shape)

print(hg)


server = Server(client_list, model, hg, features, args).to(torch.device(args.device))


loss = 0
best_sum_score = 0
best_epoch = 0
best_score = ()
for ep_index in range(args.epochs):
    for va_index in range(args.valid_step):
        t1 = time()

        sample_client = random.sample(client_list, args.batch_size)#64#采样
        server.distribute(sample_client)#model
        '''train'''
        param_list = []
        loss_list = []
        t = time()
        for idx, client in enumerate(sample_client):
            #print("yes")
            client.train()

            param, loss_c = client.train_(hg, server.user_emb, server.item_emb)
            #hg_list[client.user_id]

            param_list.append(param)  # !
            loss_list.append(loss_c.cpu())
        print(time() - t)
        #聚合参数
        server.aggregate(param_list) #!聚合参数
        loss_ = np.mean(np.array(loss_list)).item()
        loss+=loss_

        logging.info('training average loss: %.5f, time:%.1f s' % (
            loss / (ep_index * args.valid_step + va_index + 1), time() - t1))


    #test
    server.eval()
    hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10 = server.predict(test_dataloader, ep_index)
    cur_score = hit_at_5 + hit_at_10 + ndcg_at_5 + ndcg_at_10
    if(cur_score>best_sum_score):
        best_sum_score = cur_score
        best_epoch = ep_index
        best_score = (hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10)
    logging.info('Best Epoch: %d, hit_at_5 = %.4f, hit_at_10 = %.4f, ndcg_at_5 = %.4f, ndcg_at_10 = %.4f'
          % (best_epoch, best_score[0], best_score[1], best_score[2], best_score[3]))
