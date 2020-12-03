import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from tqdm import trange


class Env:
    def __init__(self):
        pygame.init()

        self.background = pygame.image.load('timg.jpg')
        self.size = self.width, self.height = self.background.get_size()

        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption('五子棋')

        self.screen.blit(self.background, [0, 0])
        self.color = [(0, 0, 0), (255, 255, 255)]
        self.alter = 0
        self.broad = np.zeros((19, 19), dtype=np.int_)
        pygame.display.flip()

    def judge(self, x, y):
        orient = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for o_x, o_y in orient:
            cnt = 0
            for i in range(-4, 0):
                now_x, now_y = x + i * o_x, y + i * o_y
                if 0 <= now_x < 19 and 0 <= now_y < 19:
                    if self.broad[now_x][now_y] == self.broad[x][y]:
                        cnt += 1
                    else:
                        break
            for i in range(0, 5):
                now_x, now_y = x + i * o_x, y + i * o_y
                if 0 <= now_x < 19 and 0 <= now_y < 19:
                    if self.broad[now_x][now_y] == self.broad[x][y]:
                        cnt += 1
                    else:
                        break
            if cnt >= 5:
                return 1
        return 0

    def event_forward(self):
        listen_event = [pygame.MOUSEBUTTONDOWN, pygame.QUIT]
        event = pygame.event.wait(0)
        while event.type not in listen_event:
            event = pygame.event.wait(0)
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if 34 <= x < 508 and 18 <= y < 500:
                pos_x = round((x - 34) / 26.33)
                pos_y = round((y - 18) / 26.33)
                self.step(pos_x, pos_y)

    def reset(self):
        self.screen.blit(self.background, [0, 0])
        self.alter = 0
        self.broad = np.zeros((19, 19), dtype=np.int_)
        pygame.display.flip()

    def step(self, pos_x, pos_y):
        if self.broad[pos_x][pos_y]:
            return None

        self.broad[pos_x][pos_y] = self.alter + 1

        fix_x = pos_x * 26.33 + 34
        fix_y = pos_y * 26.33 + 18

        pygame.draw.circle(self.screen, self.color[self.alter], (fix_x, fix_y), 12, 0)
        self.alter = 1 - self.alter
        if self.judge(pos_x, pos_y) == 1:
            font = pygame.font.SysFont('Arial', 70)
            text = font.render('GameOver', True, (242, 3, 42))
            self.screen.blit(text, ((self.width - text.get_width()) / 2, (self.height - text.get_height()) / 2))
            pygame.display.flip()
            pygame.time.wait(2000)
            pygame.event.clear()

            reward, state, over, info = (self.alter - 0.5) * 2, Image.frombytes('RGB', self.size,
                                                                                pygame.image.tostring(self.screen,
                                                                                                      'RGB')), 1, None
            self.reset()
            return reward, state, over, info
        pygame.display.flip()

        reward, state, over, info = 0, Image.frombytes('RGB', self.size,
                                                       pygame.image.tostring(self.screen, 'RGB')), 0, None
        return reward, state, over, info

    def load_map(self, map):
        self.reset()
        self.broad = map
        for x in range(0, 19):
            for y in range(0, 19):
                if map[x][y]:
                    fix_x = x * 26.33 + 34
                    fix_y = y * 26.33 + 18
                    pygame.draw.circle(self.screen, self.color[map[x][y] - 1], (fix_x, fix_y), 12, 0)
        pygame.display.flip()
        return Image.frombytes('RGB', self.size, pygame.image.tostring(self.screen, 'RGB'))


if __name__ == '__main__':
    env = Env()
    label = []
    for i in trange(5000):

        map = np.random.randint(0, 3, (19, 19))
        env.load_map(map).save('pic\\' + str(i) + '.jpg')
        label.append(map)

    label = np.dstack(label)

    label = label.transpose([2, 0, 1])
    label.dump('label.dat')
    # while True:
    #     env.event_forward()
