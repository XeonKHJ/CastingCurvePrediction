package xjtuse.castingcurvepredict.inputmodels;

public class UploadTaskLossesInputModel {
    int taskId;
    double[] losses;
    String[] times;

    public void setTimes(String[] value)
    {
        times = value;
    }

    public String[] getTimes()
    {
        return times;
    }

    public void setTaskId(int id)
    {
        this.taskId = id;
    }

    public int getTaskId()
    {
        return taskId;
    }

    public void setLosses(double[] value)
    {
        losses = value;
    }

    public double[] getLosses()
    {
        return losses;
    }
}
