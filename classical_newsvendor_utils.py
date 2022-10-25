import torch

class ClassicalNewsvendor():
    
    def __init__(self, cost_shortage, cost_excess):
        self.cs = cost_shortage
        self.ce = cost_excess
        
    def get_dist_pred_from_bnn(self, X_val, model, M):
        model.update_n_samples(n_samples=M)
        return model.forward_dist(X_val)[:,:,0]
      
    def get_argmins_from_dist(self, dist):
        quantile_cut = self.cs/(self.cs + self.ce)
        argmin_from_dist = torch.quantile(
                            dist, 
                            quantile_cut, 
                            dim=0)
        return argmin_from_dist

    def get_argmins_from_value(self, demand):
        argmin_from_value = demand
        return argmin_from_value

    def cost_per_instance(self, order, demand_true):
        return self.cs*torch.maximum(demand_true - order, torch.zeros_like(demand_true)) + self.ce*torch.maximum(order - demand_true, torch.zeros_like(demand_true))

    def cost_sum(self, order, demand_true):
        return self.cost_per_instance(order, demand_true).mean()

    def compute_norm_regret_from_costs(self, cost_pred, cost_best):
        regret =  cost_pred - cost_best
        #norm_regret = regret/cost_best
        return regret

    def compute_norm_regret(self, X_val, y_val, model, M):
        if model.output_type_dist:
            Y_pred_dist = self.get_dist_pred_from_bnn(X_val, model, M)
            z_pred = self.get_argmins_from_dist(Y_pred_dist)

        else:
            Y_pred = model(X_val)
            z_pred = self.get_argmins_from_value(Y_pred[:,0])

        z_best = self.get_argmins_from_value(y_val[:,0])

        cost_pred = self.cost_sum(z_pred, y_val[:,0])
        cost_best = self.cost_sum(z_best, y_val[:,0])

        nr = self.compute_norm_regret_from_costs(cost_pred, cost_best)

        return nr


    def compute_norm_regret_from_preds(self, X_val, y_val, Y_pred, M, method_name):
        z_pred = self.get_argmins_from_dist(Y_pred)
        z_best = self.get_argmins_from_value(y_val[:,0])

        cost_pred = self.cost_sum(z_pred, y_val[:,0])
        cost_best = self.cost_sum(z_best, y_val[:,0])

        nr = self.compute_norm_regret_from_costs(cost_pred, cost_best)
        return nr
        
    def end_loss(self, y_pred, y_true):
        z_pred = self.get_argmins_from_value(y_pred)
        end_loss = self.cost_sum(z_pred, y_true)
        return end_loss
    
    def end_loss_dist(self, y_pred_dist, y_true):
        z_pred = self.get_argmins_from_dist(y_pred_dist)
        end_loss = self.cost_sum(z_pred, y_true)
        return end_loss