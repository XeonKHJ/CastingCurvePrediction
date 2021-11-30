package xjtuse.castingcurvepredict.viewmodels;

public class TrainningStatusViewModel {
    private String _status;
    public String getStatus()
    {
        return _status;
    }

    public void setStatus(String status)
    {
        _status = status;
    }

    private int _statusCode;
    public int getStatusCode()
    {
        return _statusCode;
    }
    public void setStatusCode(int statusCode)
    {
        _statusCode = statusCode;
    }

    public double getPercentage()
    {
        return 0.5;
    }
}
