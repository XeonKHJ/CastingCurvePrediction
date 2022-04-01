package xjtuse.castingcurvepredict.viewmodels;

import xjtuse.castingcurvepredict.data.MlModel;

public class MLModelViewModel {
    public MLModelViewModel(MlModel mlmodel) {
        id = mlmodel.getId();
    }

    private int id;
    public double loss;
    private String path;

    public int getId() {
        return id;
    }
    public void setId(int value) {
        id = value;
    }

    public double getLoss() {
        return loss;
    }
    public void setLoss(double value) {
        loss = value;
    }

    

    public String getPath() {
        return path;
    }
    public void setPath(String value) {
        path = value;
    }

}
