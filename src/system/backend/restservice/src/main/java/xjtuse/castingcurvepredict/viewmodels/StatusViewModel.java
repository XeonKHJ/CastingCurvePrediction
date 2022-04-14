package xjtuse.castingcurvepredict.viewmodels;

public class StatusViewModel {
    private int _statusCode = 1;
    private String _message;
    public int getStatusCode() {
        return _statusCode;
    }

    public void setStatusCode(int statusCode) {

        _statusCode = statusCode;
    }

    public String getStatus() {
        String status;
        switch (_statusCode) {
            case 1:
                status = "成功";
                break;
            case -1:
                status = "文件为空";
                break;
            case -2:
                status = "上传失败";
                break;
            case -3:
                status = "查询不到训练活动的状态";
                break;
            default:
                status = "未知错误";
        }
        return status;
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
