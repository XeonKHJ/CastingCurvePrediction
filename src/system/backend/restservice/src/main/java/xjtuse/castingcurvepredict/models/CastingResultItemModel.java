package xjtuse.castingcurvepredict.models;

public class CastingResultItemModel {
    private String _datetime;
    private double _value;

    public CastingResultItemModel(String datetime, double value)
    {
        _datetime = datetime;
        _value = value;
    }

    public String getDatetime()
    {
        return _datetime;
    }

    public double getValue()
    {
        return _value;
    }
}
