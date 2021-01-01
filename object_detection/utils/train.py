#!/usr/bin/env python3

MODEL_PATH="../exercise_ws/src/obj_det/include/model"

def main():
    # TODO train loop here!
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

    # TODO don't forget to save the model's weights inside of f"{MODEL_PATH}/weights`!
    pass

if __name__ == "__main__":
    main()