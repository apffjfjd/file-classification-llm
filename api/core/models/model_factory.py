from api.core.models.model_handler import ModelHandler

class ModelFactory:
    @staticmethod
    def create_model_handler(model_name, model_path, hf_api_token, quantize=True):
        handler = ModelHandler(model_name, model_path, hf_api_token)
        if not handler.is_model_downloaded():
            handler.download_model(quantize=quantize)
        else:
            handler.load_model()
        return handler