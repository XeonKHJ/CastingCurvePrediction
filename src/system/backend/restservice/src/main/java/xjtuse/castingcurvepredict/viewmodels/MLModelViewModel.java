package xjtuse.castingcurvepredict.viewmodels;

import xjtuse.castingcurvepredict.data.MlModel;

public class MLModelViewModel {
    public MLModelViewModel(MlModel mlmodel)
    {
        id = mlmodel.getId();
    }

    private int id;
    public int getId()
    {
        return id;
    }
    public void setId(int value)
    {
        id = value;
    }
    
    public double loss;
    public double getLoss()
    {
        return loss;
    }
    public void setLoss(double value)
    {
        loss = value;
    }
    
}
