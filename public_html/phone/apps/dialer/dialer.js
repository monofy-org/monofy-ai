keypad = document.getElementById("keypad");
number = document.getElementById("call-number");

keypad.addEventListener("pointerdown", e=> {
    const key = e.target.getAttribute("data-key");
    if (!key) return;
      if (key == "backspace") {
        number.innerText = backspace(number.innerText);
        return;
      }
      number.innerText = formatPhoneNumber(number.innerText + key);
});

function formatPhoneNumber(input) {
    // Remove non-numeric characters from the input
    const cleanNumber = input.replace(/\D/g, '');
  
    // Format the number dynamically
    let formattedNumber = '';
  
    for (let i = 0; i < cleanNumber.length; i++) {
      // Add opening parenthesis after the first digit
      if (i === 1) {
        formattedNumber += ' (' + cleanNumber[i];
      } else if (i === 4) {
        // Add closing parenthesis and space after the fourth digit
        formattedNumber += ') ' + cleanNumber[i];
      } else if (i === 7) {
        // Add dash after the seventh digit
        formattedNumber += '-' + cleanNumber[i];
      } else {
        // Add the current character
        formattedNumber += cleanNumber[i];
      }
    }
  
    // If there are more than 10 digits, remove additional formatting
    if (cleanNumber.length > 10) {
      formattedNumber = cleanNumber;
    }
  
    return formattedNumber;
  }

  function backspace(formattedNumber) {
    // Remove non-numeric characters from the input
    const cleanNumber = formattedNumber.replace(/\D/g, '');
  
    // Remove the last digit
    const truncatedNumber = cleanNumber.slice(0, -1);
  
    // Use the existing formatting function to format the truncated number
    return formatPhoneNumber(truncatedNumber);
  }