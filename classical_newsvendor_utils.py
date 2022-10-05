import torch

class ClassicalNewsvendor():
    
    def __init__(self, sell_price, cost_price):
        self.sell_price = sell_price
        self.cost_price = cost_price
        
    def get_dist_pred_from_bnn(self, X_val, model, M):
        model.update_n_samples(n_samples=M)
        return model.forward_dist(X_val)[:,:,0]
      
    def get_argmins_from_dist(self, dist):
        quantile_cut = (self.sell_price - self.cost_price)/self.sell_price
        argmin_from_dist = torch.quantile(
                            dist, 
                            quantile_cut, 
                            dim=0)
        return argmin_from_dist

    def get_argmins_from_value(self, demand):
        argmin_from_value = demand
        return argmin_from_value

    def profit_per_instance(self, order, demand_true):
        return self.sell_price*torch.minimum(order, demand_true) - self.cost_price*order

    def profit_sum(self, order, demand_true):
        return self.profit_per_instance(order, demand_true).sum()

    def compute_norm_regret_from_profits(self, profit_pred, profit_best):
        regret = profit_best - profit_pred
        norm_regret = regret/profit_best
        return norm_regret

    def compute_norm_regret(self, X_val, y_val, model, M):
        if model.output_type_dist:
            Y_pred_dist = self.get_dist_pred_from_bnn(X_val, model, M)
            z_pred = self.get_argmins_from_dist(Y_pred_dist)

        else:
            Y_pred = model(X_val)
            z_pred = self.get_argmins_from_value(Y_pred[:,0])

        z_best = self.get_argmins_from_value(y_val[:,0])

        profit_pred = self.profit_sum(z_pred, y_val[:,0])
        profit_best = self.profit_sum(z_best, y_val[:,0])

        nr = self.compute_norm_regret_from_profits(profit_pred, profit_best)

        return nr


    def compute_norm_regret_from_preds(self, X_val, y_val, Y_pred, M, method_name):
        if method_name=='ann':
            z_pred = self.get_argmins_from_value(Y_pred[0,:])
        else:
            z_pred = self.get_argmins_from_dist(Y_pred)

        z_best = self.get_argmins_from_value(y_val[:,0])

        profit_pred = self.profit_sum(z_pred, y_val[:,0])
        profit_best = self.profit_sum(z_best, y_val[:,0])

        nr = self.compute_norm_regret_from_profits(profit_pred, profit_best)

        return nr
    
    
    def end_loss(self, y_pred, y_true):
        z_pred = self.get_argmins_from_value(y_pred)
        end_loss = -self.profit_sum(z_pred, y_true)
        return end_loss
    
    def end_loss_dist(self, y_pred_dist, y_true):
        z_pred = self.get_argmins_from_dist(y_pred_dist)
        end_loss = -self.profit_sum(z_pred, y_true)
        return end_loss