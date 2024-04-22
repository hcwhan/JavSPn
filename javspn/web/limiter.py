# A rate limiter for multithreading use case

from threading import BoundedSemaphore, Semaphore
import threading

# delay: 同一个proxy两次请求之间的间隔时间
# max_con: 最多有多少个proxy
def init_limiter(delay = 0.5, max_con: int = 1):
    BoundedSemaphore(value=max_con)
    def limiter():
        pass
    threading.Thread(limiter)
