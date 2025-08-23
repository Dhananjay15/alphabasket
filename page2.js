document.addEventListener("DOMContentLoaded", () => {
  const generateBtn = document.getElementById("generateBasket");
  const userPromptEl = document.getElementById("userPrompt");

  generateBtn.addEventListener("click", async () => {
    const userPrompt = userPromptEl.value.trim();
    const horizon = document.querySelector('input[name="horizon"]:checked')?.value;

    if (!userPrompt) {
      showError("⚠️ Please enter your investment idea before continuing.");
      return;
    }

    if (!horizon) {
      showError("⚠️ Please select an investment horizon.");
      return;
    }

    // Show loading state
    setLoading(true);

    try {
      const response = await fetch("https://alphabasket.onrender.com/generate_basket", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          prompt: userPrompt,
          horizon: horizon 
        })
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const data = await response.json();

      // ✅ store data in localStorage for Page 3
      localStorage.setItem("basketResult", JSON.stringify(data));

      // Redirect to Page 3
      window.location.href = "page3.html";
    } catch (err) {
      console.error("❌ Error generating basket:", err);
      showError("Something went wrong while generating your basket. Please try again.");
    } finally {
      setLoading(false);
    }
  });

  // 🔹 Toggle loading UI
  function setLoading(isLoading) {
    if (isLoading) {
      generateBtn.disabled = true;
      generateBtn.innerHTML = `<span class="spinner"></span> Generating...`;
    } else {
      generateBtn.disabled = false;
      generateBtn.textContent = "🚀 Generate My Basket";
    }
  }

  // 🔹 Show error inside the form
  function showError(message) {
    let errorBox = document.querySelector(".error-box");
    if (!errorBox) {
      errorBox = document.createElement("div");
      errorBox.className = "error-box glass-card";
      document.querySelector(".form-card").appendChild(errorBox);
    }
    errorBox.textContent = message;
  }
});
