package xjtuse.castingcurvepredict.viewmodels;

public class TaskViewModel {
    private String _status;
    private int _id;
    private double _loss;

    public TaskViewModel(int id, double loss, String status)
    {
        _id = id;
        _loss = loss;
        _status = status;
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
}
