import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Handler para console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

console_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)-8s | %(filename)s | %(funcName)s] (ln: %(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("pipeline.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)-8s | %(filename)s | %(funcName)s] (ln: %(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

if __name__ == "__main__":
    # Testando a configuração de logging
    logger.info("Configuração de logging inicializada com sucesso.")
    logger.debug("Esta é uma mensagem de debug.")
    logger.warning("Esta é uma mensagem de aviso.")
    logger.error("Esta é uma mensagem de erro.")
    logger.critical("Esta é uma mensagem crítica.")
