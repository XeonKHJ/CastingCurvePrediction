package xjtuse.castingcurvepredict.utils;

import java.util.Date;
import java.text.ParseException;
import java.text.SimpleDateFormat;

public class utils {

    public static String dateToString(Date date)
    {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        var str = simpleDateFormat.format(date);

        return str;
    }

    public static Date stringToDate(String dateString) throws ParseException
    {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        var date = simpleDateFormat.parse(dateString);

        return date;
    }

}
