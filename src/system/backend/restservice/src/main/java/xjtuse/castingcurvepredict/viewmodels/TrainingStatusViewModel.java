package xjtuse.castingcurvepredict.viewmodels;

public class TrainingStatusViewModel extends StatusViewModel {
    private double _percentage;
    public void setPercentage(double percentage)
    {
        _percentage = percentage;
    }

    public double getPercentage()
    {
        return _percentage;
    }
}
