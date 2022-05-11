package xjtuse.castingcurvepredict.viewmodels;

import xjtuse.castingcurvepredict.models.TaskModel;

public class TaskViewModel extends StatusViewModel {
    private String _status;
    private long _id;
    private double _loss;
    private int _epoch;
    private long _modelId;


    // 只有在出错的时候才使用该构造函数。
    public TaskViewModel(int statusCode, String message)
    {
        setStatusCode(statusCode);
        setMessage(message);
    }

    public TaskViewModel(TaskModel tm) {
        _status = tm.getStatus().toString();
        _id = tm.getId();
        _loss = tm.getLoss();
        _epoch = tm.getEpoch();
        _modelId = tm.getModelId();
    }

    public TaskViewModel(long id, double loss, String status, int epoch, int modelId) {
        _id = id;
        _loss = loss;
        _status = status;
        _epoch = epoch;
        _modelId = modelId;
    }

    public void setModelId(int value) {
        _modelId = value;
    }

    public long getModelId() {
        return _modelId;
    }

    public long getId() {
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
