package xjtuse.castingcurvepredict.viewmodels;

import xjtuse.castingcurvepredict.models.TaskModel;

public class TaskViewModel extends StatusViewModel {
    private String _status;
    private int _id;
    private double _loss;
    private int _epoch;
    private int _modelId;

    public TaskViewModel(TaskModel tm) {
        _status = tm.getStatus().toString();
        _id = (int)tm.getId();
        _loss = tm.getLoss();
        _epoch = tm.getEpoch();
    }

    public TaskViewModel(int id, double loss, String status, int epoch, int modelId) {
        _id = id;
        _loss = loss;
        _status = status;
        _epoch = epoch;
        _modelId = modelId;
    }

    public void setModelId(int value) {
        _modelId = value;
    }

    public int getModelId() {
        return _modelId;
    }

    public int getId() {
        return _id;
    }

    public double getLoss() {
        return _loss;
    }

    public String getStatus() {
        return _status;
    }

    public int getEpoch() {
        return _epoch;
    }
}
