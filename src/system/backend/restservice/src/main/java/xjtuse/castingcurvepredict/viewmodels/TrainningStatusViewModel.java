package xjtuse.castingcurvepredict.viewmodels;

import org.yaml.snakeyaml.external.com.google.gdata.util.common.base.PercentEscaper;

public class TrainningStatusViewModel extends StatusViewModel {
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
