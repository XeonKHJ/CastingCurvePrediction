var body = document.getElementsByName("body");
document.createElement()


function hasError(data)
{
    var hasError = false;
    var errorMessage;
    if(data.statusCode != undefined && data.status != undefined)
    {
        if(data.status > 0)
        {
            hasError = true;
        }
        errorMessage = data.status;
    }

    return hasError;
}

function displayError(data)
{
    document.getElementsByName("body");
    
}

