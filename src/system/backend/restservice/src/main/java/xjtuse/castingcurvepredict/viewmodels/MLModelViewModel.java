package xjtuse.castingcurvepredict.viewmodels;

public class MLModelViewModel {
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
