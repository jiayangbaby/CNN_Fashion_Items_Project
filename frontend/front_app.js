const { createApp } = Vue;

createApp({
  data() {
    return {
      selectedFile: null,
      prediction: null,
      loading: false,
      error: null,
    };
  },
  methods: {
    handleFileUpload(event) {
      this.selectedFile = event.target.files[0];
      this.prediction = null;
      this.error = null;
    },
    async predictImage() {
      if (!this.selectedFile) return;

      this.loading = true;
      this.prediction = null;
      this.error = null;

      const formData = new FormData();
      formData.append("file", this.selectedFile);

      try {
        // 1. Send image to preduction API
        const res = await fetch("http://3.133.86.70:8502/predict", {
          method: "POST",
          body: formData,
        });

        //prediction results
        let predictionResult = null;
        
        if (!res.ok) {
          const errorData = await res.json();
          this.error = errorData.error || "Prediction failed.";
        } else {
          predictionResult = await res.json();
          this.prediction = predictionResult
        }
        //2. Log user event regardless of prediction success
        await fetch ("http://3.133.86.70:8502/log",{ 
          method: "POST",
          headers: {"Content-Type": "application/json",},
          //log content
          body: JSON.stringify({
            event: "predict_clicked",
            fileName: this.selectedFile.name,
            predicted_class: predictionResult?.predicted_class || null,
            confidence: predictionResult?.confidence || null,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,}),

        });
      } catch (err) {
        this.error = "Network error: " + err.message;
      } finally {
        this.loading = false;
      }
    },
  },
}).mount("#app");
