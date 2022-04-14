package xjtuse.castingcurvepredict.viewmodels;

import javax.sound.midi.Patch;

import xjtuse.castingcurvepredict.data.MlModel;

public class MLModelViewModel {
    public MLModelViewModel(MlModel mlmodel) {
        id = mlmodel.getId();
        loss = mlmodel.getLoss();
        path = mlmodel.getPath();
        status = mlmodel.getStatus();
    }

    private int id;
    public double loss;
    private String path;
    private String status;

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

    public void setStatus(String value) {
        status = value;
    }

    public String getStatus() {
        return status;
    }

}
