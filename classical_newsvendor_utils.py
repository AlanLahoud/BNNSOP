import torch

# Class to the Classical Newsvendor Optimization Problem
class ClassicalNewsvendor():
    """
    Classical Newsvendor Optimization Problem class.
    Init with c_s and c_e deterministic parameters.
    """
    
    def __init__(self, cost_shortage, cost_excess):
        self.cs = cost_shortage
        self.ce = cost_excess
      
    def get_argmins_from_dist(self, dist):
        """
        Give samples of y: dist, compute z*(dist)
        """
        quantile_cut = self.cs/(self.cs + self.ce)
        argmin_from_dist = torch.quantile(
                            dist, 
                            quantile_cut, 
                            dim=0)
        argmin_from_dist = torch.maximum(
            argmin_from_dist, 
            torch.zeros_like(argmin_from_dist))
        return argmin_from_dist

    def get_argmins_from_value(self, demand):
        """
        Give values of y, compute z*(y) ( = y in this case)
        """
        argmin_from_value = argmin_from_dist = torch.maximum(
            demand, torch.zeros_like(demand))
        return argmin_from_value

    def cost_per_instance(self, order, demand_true):
        """
        Give values of z and y, compute the cost f(z,y) per row
        """
        return self.cs*torch.maximum(
            demand_true - order, 
            torch.zeros_like(demand_true)) + self.ce*torch.maximum(
            order - demand_true, torch.zeros_like(demand_true))

    def cost_sum(self, order, demand_true):
        """
        Compute mean of f(z, y) through rows
        """
        return self.cost_per_instance(order, demand_true).mean()

    def compute_norm_regret_from_costs(self, cost_pred, cost_best):
        regret =  cost_pred - cost_best
        return regret

    def compute_norm_regret_from_preds(self, y_val, Y_pred, Y_noisy):
        """
        Compute evaluation metrics regret and fair regret
        """
        z_pred = self.get_argmins_from_dist(Y_pred)
        z_fair = self.get_argmins_from_dist(Y_noisy)
        z_best = self.get_argmins_from_value(y_val[:,0])

        cost_pred = self.cost_sum(z_pred, y_val[:,0])
        cost_fair = self.cost_sum(z_fair, y_val[:,0])
        cost_best = self.cost_sum(z_best, y_val[:,0])

        reg = self.compute_norm_regret_from_costs(cost_pred, cost_best)
        freg = self.compute_norm_regret_from_costs(cost_pred, cost_fair)
        return reg, freg
        
    def end_loss(self, y_pred, y_true):
        """
        Compute cost function, used in ANN combined 
        backpropagation
        """
        z_pred = self.get_argmins_from_value(y_pred)
        end_loss = self.cost_sum(z_pred, y_true)
        return end_loss
    
    def end_loss_dist(self, y_pred_dist, y_true):
        """
        Compute cost function given samples, used in BNN 
        combined backpropagation
        """
        z_pred = self.get_argmins_from_dist(y_pred_dist)
        end_loss = self.cost_sum(z_pred, y_true)
        return end_loss