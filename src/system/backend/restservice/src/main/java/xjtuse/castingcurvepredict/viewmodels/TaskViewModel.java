package xjtuse.castingcurvepredict.viewmodels;

public class TaskViewModel {
    private String _status;
    private int _id;
    private double _loss;
    private int _epoch;

    public TaskViewModel(int id, double loss, String status, int epoch) {
        _id = id;
        _loss = loss;
        _status = status;
        _epoch = epoch;
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

    public int getEpoch()
    {
        return _epoch;
    }
}
