package xjtuse.castingcurvepredict.models;

import org.apache.tomcat.util.threads.StopPooledThreadException;

public class CastingResultItemModel {
    private String _datetime;
    private double _liqLevel;
    private double _stopperPos;

    public CastingResultItemModel(String datetime, double stopperPos, double liqLevel )
    {
        _datetime = datetime;
        _liqLevel = liqLevel;
        _stopperPos = stopperPos;
    }

    public CastingResultItemModel(String datetime, double stopperPos)
    {
        _datetime = datetime;
        _stopperPos = stopperPos;
    }

    public String getDatetime()
    {
        return _datetime;
    }

    public double getStopperPos()
    {
        return _stopperPos;
    }

    public double getLiqLevel()
    {
        return _liqLevel;
    }
}
