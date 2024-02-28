from torch.nn.parameter import Parameter
from model import *
from eval import *
from util import *

class Server(nn.Module):
    def __init__(self, client_list, model, hg, features, args):
        super().__init__()
        self.device = args.device
        self.hg = hg
        self.client_list = client_list
        self.features = features
        self.model_user = model[0]#(0:model_user, 1: model_item)
        self.model_item = model[1]
        self.user_emb = nn.Embedding(features[0].shape[0], features[0].shape[1]).to(self.device)
        self.item_emb = nn.Embedding(features[1].shape[0], features[1].shape[1]).to(self.device)
        self.user_emb.weight.data = Parameter(torch.Tensor(features[0])).to(self.device)
        self.item_emb.weight.data = Parameter(torch.Tensor(features[1])).to(self.device)
        #nn.init.normal_(self.item_emb.weight, std=0.01)
        self.lr = args.lr
        self.weight_decay = args.weight_decay



    def aggregate(self, param_list):
        flag = False
        number = 0
        gradient_item = torch.zeros_like(self.item_emb.weight)
        gradient_user = torch.zeros_like(self.user_emb.weight)
        item_count = torch.zeros(self.item_emb.weight.shape[0]).to(self.device)
        user_count = torch.zeros(self.user_emb.weight.shape[0]).to(self.device)

        for parameter in param_list:
            model_grad_user, model_grad_item = parameter['model']
            item_grad, returned_items = parameter['item']
            user_grad, returned_users = parameter['user']
            num = len(returned_items)
            item_count[returned_items] += 1
            user_count[returned_users] += num

            number += num
            if not flag:
                flag = True
                gradient_model_user = []
                gradient_model_item = []
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad_user)):
                    gradient_model_user.append(model_grad_user[i]* num)
                for i in range(len(model_grad_item)):
                    gradient_model_item.append(model_grad_item[i]* num)
            else:
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad_user)):
                    gradient_model_user[i] += model_grad_user[i] * num
                for i in range(len(model_grad_item)):
                    gradient_model_item[i] += model_grad_item[i] * num

        item_count[item_count == 0] = 1
        user_count[user_count == 0] = 1
        gradient_item /= item_count.unsqueeze(1)
        gradient_user /= user_count.unsqueeze(1)
        for i in range(len(gradient_model_user)):
            gradient_model_user[i] = gradient_model_user[i] / number
        for i in range(len(gradient_model_item)):
            gradient_model_item[i] = gradient_model_item[i] / number


        #更新model参数
        ls_model_param_user = list(self.model_user.parameters())
        ls_model_param_item = list(self.model_item.parameters())
        for i in range(len(ls_model_param_user)):
            ls_model_param_user[i].data = ls_model_param_user[i].data - self.lr * gradient_model_user[i] - self.weight_decay * ls_model_param_user[i].data
        for i in range(len(ls_model_param_item)):
            ls_model_param_item[i].data = ls_model_param_item[i].data - self.lr * gradient_model_item[i] - self.weight_decay * ls_model_param_item[i].data

        # for i in range(len(list(self.model_user.parameters()))):
        #     print(ls_model_param_user[i].data)
        #     break
        #更新item/user参数
        item_index = gradient_item.sum(dim = -1) != 0
        user_index = gradient_user.sum(dim = -1) != 0
        with torch.no_grad():#不加会报错
            self.item_emb.weight[item_index] = self.item_emb.weight[item_index] -  self.lr * gradient_item[item_index] - self.weight_decay * self.item_emb.weight[item_index]
            self.user_emb.weight[user_index] = self.user_emb.weight[user_index] -  self.lr * gradient_user[user_index] - self.weight_decay * self.user_emb.weight[user_index]



    def distribute(self, client_list):
        for client in client_list:
            client.update(self.model_user, self.model_item)


    def predict(self, test_dataloader, epoch):
        hit_at_5 = []
        hit_at_10 = []
        ndcg_at_5 = []
        ndcg_at_10 = []

        self.model_item.eval()
        self.model_user.eval()
        logits_user = self.model_user(self.hg, self.user_emb.weight)
        logits_item = self.model_item(self.hg, self.item_emb.weight)
        for u, i, neg_i in test_dataloader: #test_i算上了test_negative, 真实的放在最后一位[99]
            cur_user = logits_user[u]
            cur_item = logits_item[i]
            rating = torch.sum(cur_user * cur_item, dim=-1)#当前client user和所有item点乘(include test item)

            for eva_idx, eva in enumerate(rating):
                cur_neg = logits_item[neg_i[eva_idx]]
                cur_rating_neg = torch.sum(cur_user[eva_idx] * cur_neg, dim=-1)
                #print(np.shape(cur_rating_neg))
                cur_eva = torch.cat([cur_rating_neg, torch.unsqueeze(rating[eva_idx], 0)], dim=0)
                #print(np.shape(rating[eva_idx]))
                # print(cur_eva)
                hit_at_5_ = evaluate_recall(cur_eva, [99], 5)#[99]是测试集(ground truth)
                hit_at_10_ = evaluate_recall(cur_eva, [99], 10)
                ndcg_at_5_ = evaluate_ndcg(cur_eva, [99], 5)
                ndcg_at_10_ = evaluate_ndcg(cur_eva, [99], 10)
                #print(hit_at_10_)
                hit_at_5.append(hit_at_5_)
                hit_at_10.append(hit_at_10_)
                ndcg_at_5.append(ndcg_at_5_)
                ndcg_at_10.append(ndcg_at_10_)
        hit_at_5 = np.mean(np.array(hit_at_5)).item()
        hit_at_10 = np.mean(np.array(hit_at_10)).item()
        ndcg_at_5 = np.mean(np.array(ndcg_at_5)).item()
        ndcg_at_10 = np.mean(np.array(ndcg_at_10)).item()

        logging.info('Epoch: %d, hit_at_5 = %.4f, hit_at_10 = %.4f, ndcg_at_5 = %.4f, ndcg_at_10 = %.4f'
              % (epoch, hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10))
        return hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10










