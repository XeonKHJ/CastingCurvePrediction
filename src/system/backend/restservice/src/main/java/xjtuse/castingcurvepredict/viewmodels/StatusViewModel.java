package xjtuse.castingcurvepredict.viewmodels;

public class StatusViewModel {
    private int _statusCode = 0;
    private String _status;
    private String _message;
    public int getStatusCode() {
        return _statusCode;
    }

    public void setStatusCode(int statusCode) {
        switch (statusCode) {
            case 0:
                _status = "成功";
            case -1:
                _status = "文件为空";
                break;
            case -2:
                _status = "上传失败";
                break;
            case -3:
                _status = "查询不到训练活动的状态";
            default:
                _status = "未知错误";
        }
        _statusCode = statusCode;
    }

    public String getStatus() {
        return _status;
    }

    public void setMessage(String message)
    {
        _message = message;
    }

    public String getMessage()
    {
        return _message;
    }
}
