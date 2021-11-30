var body = document.getElementsByName("body");
document.createElement()


function DisplayError(data)
{
    var hasError = false;
    var errorMessage;
    if(data.statusCode != undefined && data.status != undefined)
    {
        if(data.status != 0)
        {
            hasError = true;
        }
        errorMessage = data.status;
    }

    return hasError;
}